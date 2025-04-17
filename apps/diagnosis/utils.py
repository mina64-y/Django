# apps/diagnosis/utils.py

import os
import uuid
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use('Agg') # GUI 없는 환경용 백엔드 설정
import matplotlib.pyplot as plt
from django.conf import settings
from django.http import HttpResponse # for PDF response
from django.template.loader import render_to_string
from django.core.files.storage import default_storage # 파일 시스템 접근용
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, Resized, EnsureTyped
    # Activations, AsDiscrete 등 후처리 관련 transform 은 task 에서 직접 사용하거나 별도 함수로 분리 가능
)
import traceback
from collections import OrderedDict
import google.generativeai as genai # Gemini
from weasyprint import HTML, CSS # Weasyprint for PDF
import plotly.graph_objects as go # Plotly for 3D
from skimage.measure import marching_cubes # Marching cubes for 3D mesh

# --- Gemini 설정 ---
GEMINI_API_KEY = settings.GEMINI_API_KEY
GEMINI_CONFIGURED = False
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest") # 모델 이름 확인/변경 가능
        GEMINI_CONFIGURED = True
        print("Gemini AI configured successfully for CT diagnosis.")
    except Exception as e:
        print(f"!!!!!!!! ERROR configuring Gemini AI !!!!!!!!!\n{e}")
        traceback.print_exc()
else:
    print("WARN: GEMINI_API_KEY not found in settings. CT Gemini interpretation disabled.")

# --- 전역 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOADED_PYTORCH_MODELS = {} # 간단한 모델 캐싱용 딕셔너리

# --- 모델별 전처리 파라미터 정의 ---
CLASSIFICATION_SPATIAL_SHAPE = (96, 96, 96) # (D, H, W)
CLASSIFICATION_SPACING = (1.5, 1.5, 2.0)
CLASSIFICATION_HU_WINDOW = (-57, 164)
CLASSIFICATION_ORIENTATION = "RAS"

SEGMENTATION_SPATIAL_SHAPE = (64, 96, 96) # (D, H, W)
SEGMENTATION_HU_WINDOW = (-100, 240)
SEGMENTATION_ORIENTATION = "RAS"

# --- 모델 클래스 정의 ---

class Simple3DClassifier(nn.Module):
    """3D CNN 분류 모델 (분류 학습 스크립트 버전)"""
    def __init__(self, in_channels=1, num_classes=1, channels=None, strides=None, fc_dims=None, dropout=0.2):
        super().__init__()
        # 파라미터 기본값 설정
        if channels is None: channels = [16, 32, 64, 128]
        if strides is None: strides = [(2, 2, 2)] * len(channels)
        if fc_dims is None: fc_dims = [256]
        if len(channels) != len(strides):
            raise ValueError("Length of channels and strides must match")

        self.encoder = nn.Sequential()
        current_channels = in_channels
        for i, (ch, st) in enumerate(zip(channels, strides)):
            self.encoder.add_module(f"conv{i+1}", nn.Conv3d(current_channels, ch, kernel_size=3, stride=st, padding=1, bias=False))
            self.encoder.add_module(f"bn{i+1}", nn.BatchNorm3d(ch))
            self.encoder.add_module(f"relu{i+1}", nn.ReLU(inplace=True))
            current_channels = ch

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential()
        in_features = current_channels
        for i, hidden_dim in enumerate(fc_dims):
            self.classifier.add_module(f"fc{i+1}", nn.Linear(in_features, hidden_dim))
            self.classifier.add_module(f"fc_relu{i+1}", nn.ReLU(inplace=True))
            if dropout > 0:
                self.classifier.add_module(f"fc_dropout{i+1}", nn.Dropout(dropout))
            in_features = hidden_dim
        self.classifier.add_module("fc_out", nn.Linear(in_features, num_classes))

    def forward(self, x):
        x = self.encoder(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

def get_activation(activation_type, negative_slope=0.01):
    """ 활성화 함수 반환 (분할 학습 스크립트 버전) """
    if activation_type.lower() == 'relu': return nn.ReLU(inplace=True)
    elif activation_type.lower() == 'leaky_relu': return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
    else: raise ValueError(f"Unsupported activation function type: {activation_type}")

class Custom3DUNet(nn.Module):
    """3D U-Net 분할 모델 (분할 학습 스크립트와 동일한 구조)"""
    def __init__(self, in_channels=1, out_channels=3, filters=(16, 32, 64, 128), dropout_rate=0.15, activation='leaky_relu', leaky_slope=0.01):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.leaky_slope = leaky_slope

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        def conv_block(ic, oc, act, ls, dr):
            layers = [
                nn.Conv3d(ic, oc, kernel_size=3, padding=1, bias=False), nn.BatchNorm3d(oc), get_activation(act, ls),
                nn.Conv3d(oc, oc, kernel_size=3, padding=1, bias=False), nn.BatchNorm3d(oc), get_activation(act, ls),
            ]
            if dr > 0.0: layers.append(nn.Dropout3d(dr))
            return nn.Sequential(*layers)

        current_channels = in_channels
        for i, f in enumerate(filters):
            current_dropout = self.dropout_rate * (i / (len(filters) - 1)) if len(filters) > 1 and self.dropout_rate > 0 else 0.0
            encoder = conv_block(current_channels, f, self.activation, self.leaky_slope, current_dropout)
            pool = nn.MaxPool3d(kernel_size=2, stride=2) if i < len(filters) - 1 else nn.Identity()
            self.encoders.append(encoder)
            self.pools.append(pool)
            current_channels = f

        bn_filters = filters[-1] * 2
        self.bottleneck = conv_block(current_channels, bn_filters, self.activation, self.leaky_slope, dropout_rate)

        current_channels = bn_filters
        reversed_filters = list(reversed(filters))
        for i, f in enumerate(reversed_filters):
            upconv = nn.ConvTranspose3d(current_channels, f, kernel_size=2, stride=2)
            self.upconvs.append(upconv)
            concat_channels = f + f # Skip connection channel + Upconv channel
            current_dropout = self.dropout_rate * ((len(filters) - 1 - i) / (len(filters) - 1)) if len(filters) > 1 and self.dropout_rate > 0 else 0.0
            decoder = conv_block(concat_channels, f, self.activation, self.leaky_slope, current_dropout)
            self.decoders.append(decoder)
            current_channels = f

        self.output_conv = nn.Conv3d(current_channels, self.out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        current_feature = x
        # Encoder path
        for i in range(len(self.encoders)):
            current_feature = self.encoders[i](current_feature)
            skips.append(current_feature)
            current_feature = self.pools[i](current_feature) # Apply pooling (or Identity for the last one)

        # Bottleneck
        current_feature = self.bottleneck(current_feature)

        # Decoder path
        skips = list(reversed(skips)) # Reverse for decoder skip connections

        for i in range(len(self.decoders)):
            current_feature = self.upconvs[i](current_feature) # Upsample

            skip_connection = skips[i] # Get corresponding skip connection

            # Handle shape mismatch (Padding/Cropping - MONAI's Pad or Crop might be more robust if needed)
            if current_feature.shape[2:] != skip_connection.shape[2:]:
                target_shape = current_feature.shape[2:]
                skip_shape = skip_connection.shape[2:]
                padding = []
                cropping = []
                needs_padding = False
                needs_cropping = False
                for d in range(3): # D, H, W
                    diff = target_shape[d] - skip_shape[d]
                    if diff > 0: needs_padding = True; pad_before = diff//2; pad_after = diff-pad_before; padding.extend([pad_before, pad_after]); cropping.extend([0,0])
                    elif diff < 0: needs_cropping = True; crop_before = -diff//2; crop_after = -diff-crop_before; cropping.extend([crop_before, crop_after]); padding.extend([0,0])
                    else: padding.extend([0,0]); cropping.extend([0,0])

                if needs_padding:
                    pad_params = (padding[4], padding[5], padding[2], padding[3], padding[0], padding[1]) # W, H, D
                    if all(p>=0 for p in pad_params): skip_connection = F.pad(skip_connection, pad_params)
                    else: raise RuntimeError(f"Cannot pad skip connection {skip_connection.shape} to {current_feature.shape}")
                elif needs_cropping:
                    crop_slices = tuple(slice(cropping[d*2], skip_shape[d]-cropping[d*2+1]) for d in range(3))
                    skip_connection = skip_connection[(slice(None), slice(None)) + crop_slices] # Crop C, D, H, W

            # Concatenate skip connection and upsampled feature map
            current_feature = torch.cat((skip_connection, current_feature), dim=1)

            # Apply decoder block
            current_feature = self.decoders[i](current_feature)

        # Final output layer
        output = self.output_conv(current_feature)
        return output

# --- 전처리 파이프라인 ---
def get_classification_transforms():
    """ CT 분류용 전처리 변환 """
    keys = ["image"]
    return Compose(
        [
            LoadImaged(keys=keys, image_only=True, ensure_channel_first=True),
            Spacingd(keys=keys, pixdim=CLASSIFICATION_SPACING, mode="bilinear", align_corners=True),
            Orientationd(keys=keys, axcodes=CLASSIFICATION_ORIENTATION),
            ScaleIntensityRanged(keys=keys, a_min=CLASSIFICATION_HU_WINDOW[0], a_max=CLASSIFICATION_HU_WINDOW[1],
                                 b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=keys, source_key="image", allow_smaller=True, margin=10),
            Resized(keys=keys, spatial_size=CLASSIFICATION_SPATIAL_SHAPE, mode="area"), # Resize to target shape
            EnsureTyped(keys=keys, dtype=torch.float32),
        ]
    )

def get_segmentation_transforms():
    """ CT 분할용 전처리 변환 """
    keys = ["image"]
    return Compose(
        [
            LoadImaged(keys=keys, image_only=True, ensure_channel_first=True),
            # Spacingd(keys=keys, pixdim=(1.0, 1.0, 1.0)), # 학습 시 사용했다면 추가
            Orientationd(keys=keys, axcodes=SEGMENTATION_ORIENTATION),
            ScaleIntensityRanged(keys=keys, a_min=SEGMENTATION_HU_WINDOW[0], a_max=SEGMENTATION_HU_WINDOW[1],
                                 b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=keys, source_key="image", allow_smaller=True, margin=10),
            Resized(keys=keys, spatial_size=SEGMENTATION_SPATIAL_SHAPE, mode="area"),
            EnsureTyped(keys=keys, dtype=torch.float32),
        ]
    )

# --- 모델 로드 함수 ---
def load_pytorch_model(model_key, model_path, model_class, model_params):
    """ 키를 사용하여 모델을 로드하고 캐싱합니다. """
    # 캐시 확인 (Celery 환경에서는 프로세스마다 캐시될 수 있으므로 주의)
    # 더 강력한 캐싱이 필요하면 Redis나 파일 기반 캐시 고려
    if model_key in LOADED_PYTORCH_MODELS and LOADED_PYTORCH_MODELS[model_key] is not None:
         print(f"Using cached model '{model_key}'")
         return LOADED_PYTORCH_MODELS[model_key]

    if model_path and os.path.exists(model_path):
        try:
            print(f"Loading model '{model_key}' from {model_path}...")
            model_instance = model_class(**model_params)
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False) # 보안 경고 무시

            state_dict = None
            # state_dict 추출 로직 (다양한 저장 방식 처리)
            if isinstance(checkpoint, OrderedDict): state_dict = checkpoint
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
            # 모델 객체 자체인 경우 (덜 일반적)
            elif isinstance(checkpoint, nn.Module): state_dict = checkpoint.state_dict()
            else: # 키가 직접 있는 경우 추정
                 if checkpoint is not None and hasattr(checkpoint, 'keys') and callable(getattr(checkpoint, 'keys')):
                     if any(k.startswith('encoder.') or k.startswith('classifier.') for k in checkpoint.keys()): # Simple3DClassifier 키 예시
                        state_dict = checkpoint
                     elif any(k.startswith('encoders.') or k.startswith('bottleneck.') or k.startswith('decoders.') for k in checkpoint.keys()): # Custom3DUNet 키 예시
                        state_dict = checkpoint

            if state_dict:
                # 'module.' 접두사 제거 (DataParallel 등으로 학습된 경우)
                first_key = next(iter(state_dict))
                if first_key.startswith('module.'):
                    print("Removing 'module.' prefix from state_dict keys...")
                    state_dict = OrderedDict((k[len("module."):], v) for k, v in state_dict.items())

                model_instance.load_state_dict(state_dict)
                model_instance.to(DEVICE)
                model_instance.eval()
                LOADED_PYTORCH_MODELS[model_key] = model_instance # 캐시에 저장
                print(f"Successfully loaded model '{model_key}' to {DEVICE}")
                return model_instance
            else:
                print(f"ERROR: Could not extract state_dict from checkpoint for '{model_key}'. Checkpoint type: {type(checkpoint)}")
                LOADED_PYTORCH_MODELS[model_key] = None # 로드 실패 기록
        except Exception as e:
            print(f"!!!!!!!! ERROR loading model '{model_key}' from {model_path} !!!!!!!!!")
            print(f"Error details: {e}"); traceback.print_exc()
            LOADED_PYTORCH_MODELS[model_key] = None
    else:
        print(f"ERROR: Model file not found or path invalid for '{model_key}': {model_path}")
        LOADED_PYTORCH_MODELS[model_key] = None

    return None # 최종적으로 실패 시 None 반환

# --- 각 모델 인스턴스 가져오기 ---
def get_pancreas_classifier():
    """분류 모델 인스턴스를 로드하거나 캐시에서 가져옵니다."""
    model_params = {
        'in_channels': 1, 'num_classes': 1, 'channels': [16, 32, 64, 128],
        'strides': [(2, 2, 2)] * 4, 'fc_dims': [256], 'dropout': 0.2
    }
    model_path = getattr(settings, 'PANCREAS_CLASSIFY_MODEL_PATH', None)
    return load_pytorch_model('pancreas_classify', model_path, Simple3DClassifier, model_params)

def get_pancreas_segmenter():
    """분할 모델 인스턴스를 로드하거나 캐시에서 가져옵니다."""
    model_params = {
        'in_channels': 1, 'out_channels': 3, # 배경, 췌장, 종양 = 3 클래스
        'filters': getattr(settings, 'CUSTOM_UNET_FILTERS', [16, 32, 64, 128]),
        'dropout_rate': getattr(settings, 'CUSTOM_UNET_DROPOUT', 0.15),
        'activation': getattr(settings, 'CUSTOM_UNET_ACTIVATION', 'leaky_relu'),
        'leaky_slope': getattr(settings, 'CUSTOM_UNET_LEAKY_SLOPE', 0.01)
    }
    model_path = getattr(settings, 'PANCREAS_SEGMENT_MODEL_PATH', None)
    # 모델 클래스는 Custom3DUNet으로 가정 (설정에서 변경 가능하게 하려면 추가 로직 필요)
    return load_pytorch_model('pancreas_segment', model_path, Custom3DUNet, model_params)

# --- 추론 및 후처리 함수 ---
def run_inference(model, image_tensor):
    """단일 이미지 텐서에 대해 모델 추론 실행"""
    if model is None or image_tensor is None:
        print("ERROR: Model or image tensor is None for inference.")
        return None
    try:
        model.eval() # 평가 모드 확인
        with torch.no_grad():
            output = model(image_tensor) # 입력 텐서는 이미 device에 있어야 함
        return output
    except Exception as e:
        print(f"!!!!!!!! ERROR during model inference !!!!!!!!!")
        print(f"Model type: {type(model)}, Input tensor shape: {image_tensor.shape}, Device: {image_tensor.device}")
        print(f"Error details: {e}"); traceback.print_exc()
        return None

def postprocess_pancreas_classification(output_tensor):
    """분류 모델 출력(logit)을 확률과 예측 레이블로 변환"""
    if output_tensor is None: return None, None
    try:
        # 출력이 단일 logit 값이라고 가정 (B, 1)
        probability = torch.sigmoid(output_tensor).item() # 배치 차원 제거 후 스칼라 값으로
        prediction_label = 1 if probability >= 0.5 else 0 # 임계값 0.5 기준
        return probability, prediction_label
    except Exception as e:
        print(f"!!!!!!!! ERROR during classification postprocessing !!!!!!!!!")
        print(f"Input tensor shape: {output_tensor.shape}")
        print(f"Error details: {e}"); traceback.print_exc()
        return None, None

def postprocess_pancreas_segmentation(output_tensor):
    """분할 모델 출력(logits)을 최종 마스크(NumPy)로 변환"""
    if output_tensor is None: return None
    try:
        # MONAI의 후처리 변환 사용 가능 (Activations, AsDiscrete)
        # 예시: 직접 argmax 사용
        # 출력 shape: (B, NumClasses, D, H, W)
        pred_mask_tensor = torch.argmax(output_tensor, dim=1, keepdim=False) # (B, D, H, W)
        # 배치 차원 제거 및 CPU로 이동, NumPy 배열로 변환
        pred_mask_np = pred_mask_tensor.squeeze(0).cpu().numpy().astype(np.uint8) # (D, H, W)
        return pred_mask_np
    except Exception as e:
        print(f"!!!!!!!! ERROR during segmentation postprocessing !!!!!!!!!")
        print(f"Input tensor shape: {output_tensor.shape}")
        print(f"Error details: {e}"); traceback.print_exc()
        return None

def get_segmented_volume_voxels(mask_np, target_class=2):
    """ 분할 마스크에서 특정 클래스 복셀 수 계산 (종양=2 가정) """
    if mask_np is None or not isinstance(mask_np, np.ndarray): return 0
    try:
        volume_voxels = np.sum(mask_np == target_class)
        return int(volume_voxels)
    except Exception as e:
        print(f"!!!!!!!! ERROR calculating segmented volume !!!!!!!!!")
        print(f"Mask shape: {mask_np.shape}, Target class: {target_class}")
        print(f"Error details: {e}"); traceback.print_exc()
        return 0

# --- 시각화 함수 ---
def save_plot_to_media(plot_func, filename_prefix, plot_args=None, plot_kwargs=None):
    """ Matplotlib 플롯 함수를 실행하고 결과를 MEDIA 경로에 저장, 상대 경로 반환 """
    if plot_args is None: plot_args = []
    if plot_kwargs is None: plot_kwargs = {}
    try:
        # 플롯 함수는 figure 객체를 반환해야 함
        fig = plot_func(*plot_args, **plot_kwargs)
        if fig is None:
            print(f"Plot function for '{filename_prefix}' returned None.")
            return None

        # 파일명 및 저장 경로 설정 (media/plots/diagnosis/)
        filename = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.png"
        save_dir_rel = os.path.join('plots', 'diagnosis') # 상대 경로
        save_dir_abs = os.path.join(settings.MEDIA_ROOT, save_dir_rel)
        os.makedirs(save_dir_abs, exist_ok=True) # 폴더 생성
        save_path_abs = os.path.join(save_dir_abs, filename)

        # 저장
        fig.savefig(save_path_abs, bbox_inches='tight', dpi=100)
        plt.close(fig) # 메모리 해제
        print(f"Saved plot: {save_path_abs}")

        # MEDIA_ROOT 기준 상대 경로 반환
        return os.path.join(save_dir_rel, filename).replace("\\", "/")

    except Exception as e:
        print(f"!!!!!!!! ERROR saving plot '{filename_prefix}' !!!!!!!!!")
        print(traceback.format_exc())
        if 'fig' in locals() and isinstance(fig, plt.Figure): plt.close(fig) # 오류 시 플롯 닫기
        return None

# --- 2D 플롯 생성 함수 (fig 반환) ---
def generate_nifti_slice_plot(nifti_abs_path, slice_dim=2, title="Input Slice"):
    """ NIfTI 파일의 중간 슬라이스 플롯 생성 (fig 반환) """
    try:
        img = nib.load(nifti_abs_path)
        data = img.get_fdata(dtype=np.float32, caching='unchanged')
        if data.ndim != 3: raise ValueError(f"3D data expected, got {data.ndim}D")

        slice_idx = data.shape[slice_dim] // 2
        if slice_dim == 0: slice_data = data[slice_idx, :, :]
        elif slice_dim == 1: slice_data = data[:, slice_idx, :]
        else: slice_data = data[:, :, slice_idx] # Default Z

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.rot90(slice_data), cmap='gray', aspect='auto')
        ax.set_title(f"{title} (Dim {slice_dim}, Idx {slice_idx})", fontsize=10)
        ax.axis('off')
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error generating NIfTI slice plot: {e}")
        traceback.print_exc()
        return None

def generate_segmentation_overlay_plot(nifti_abs_path, mask_np, slice_dim=2, title="Segmentation Overlay"):
    """ 분할 마스크 오버레이 플롯 생성 (fig 반환) """
    try:
        img_nib = nib.load(nifti_abs_path)
        img_data = img_nib.get_fdata(dtype=np.float32, caching='unchanged')
        if img_data.ndim != 3: raise ValueError("Image must be 3D")
        if mask_np is None or mask_np.ndim != 3:
             print("WARN: Valid mask not provided for overlay plot.")
             # 마스크 없으면 원본 슬라이스만 반환
             return generate_nifti_slice_plot(nifti_abs_path, slice_dim, title="Input (No Segmentation)")

        mask_to_plot = mask_np.astype(np.uint8) # 타입 변환

        # 리사이징 필요 시 (원본 이미지와 마스크 크기가 다를 때 - 비권장)
        if img_data.shape != mask_to_plot.shape:
            print(f"WARN: Resizing mask shape {mask_to_plot.shape} to image shape {img_data.shape} for overlay. Result might be inaccurate.")
            resizer = Resized(keys=['mask'], spatial_size=img_data.shape, mode="nearest")
            mask_resized = resizer({'mask': mask_to_plot[np.newaxis, ...]})['mask'].squeeze(0) # Add/Remove channel dim
            if mask_resized.shape != img_data.shape: raise RuntimeError("Mask resize failed")
            mask_to_plot = mask_resized

        # 슬라이스 추출
        slice_idx = img_data.shape[slice_dim] // 2
        if slice_dim == 0: img_slice=img_data[slice_idx,:,:]; mask_slice=mask_to_plot[slice_idx,:,:]
        elif slice_dim == 1: img_slice=img_data[:,slice_idx,:]; mask_slice=mask_to_plot[:,slice_idx,:]
        else: img_slice=img_data[:,:,slice_idx]; mask_slice=mask_to_plot[:,:,slice_idx] # Default Z

        # 플롯 생성
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.rot90(img_slice), cmap='gray', aspect='auto') # 원본 이미지

        # 마스크 오버레이 (췌장=하늘색, 종양=빨간색 가정)
        colors = {1: 'cyan', 2: 'red'} # 레이블 1: 췌장, 레이블 2: 종양
        alphas = {1: 0.3, 2: 0.5}
        for cls_val in np.unique(mask_slice):
            if cls_val == 0: continue # 배경 제외
            color = colors.get(cls_val, 'white') # 정의되지 않은 레이블은 흰색
            alpha = alphas.get(cls_val, 0.4)
            masked_data = np.ma.masked_where(mask_slice != cls_val, mask_slice)
            ax.imshow(np.rot90(masked_data), cmap=matplotlib.colors.ListedColormap([color]),
                      alpha=alpha, aspect='auto', interpolation='nearest')

        ax.set_title(f"{title} (Dim {slice_dim}, Idx {slice_idx})", fontsize=10)
        ax.axis('off')
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error generating segmentation overlay plot: {e}")
        traceback.print_exc()
        # 오류 시 원본 슬라이스 플롯이라도 반환 시도
        return generate_nifti_slice_plot(nifti_abs_path, slice_dim, title="Input (Overlay Error)")

# --- 3D 시각화 이미지 생성 함수 ---
def generate_3d_visualization_html(mask_np, filename_prefix="ct_3d_vis_html", # 이름 변경
                                   label_pancreas=1, label_tumor=2,
                                   step_size=2, pancreas_color='rgb(107, 174, 214)',
                                   tumor_color='rgb(255, 0, 0)', opacity=0.5):
    """ 3D 메쉬 생성 후 Plotly HTML 파일로 저장하고 상대 경로 반환 """
    print("\nGenerating 3D visualization HTML...")
    if mask_np is None or not isinstance(mask_np, np.ndarray):
        print("WARN: Invalid mask provided for 3D visualization HTML.")
        return None

    mask_np_uint8 = mask_np.astype(np.uint8)
    plotly_data = []
    labels_info = {'Pancreas': {'label': label_pancreas, 'color': pancreas_color},
                   'Tumor': {'label': label_tumor, 'color': tumor_color}}

    for name, info in labels_info.items():
        label_val, color = info['label'], info['color']
        current_mask = (mask_np_uint8 == label_val)
        if np.any(current_mask):
            print(f"  Generating mesh for {name}...")
            try:
                padded_mask = np.pad(current_mask, pad_width=1, mode='constant', constant_values=0)
                verts, faces, _, _ = marching_cubes(padded_mask, level=0.5, step_size=step_size)
                verts = verts - 1
                x, y, z = verts.T; i, j, k = faces.T
                mesh = go.Mesh3d(x=z, y=y, z=x, i=k, j=j, k=i, color=color, opacity=opacity, name=name)
                plotly_data.append(mesh)
            except Exception as e: print(f"Error generating mesh for {name}: {e}"); traceback.print_exc()
        else: print(f"  No voxels found for {name}.")

    if not plotly_data:
        print("No meshes generated, skipping HTML file creation.")
        return None

    # Plotly Figure 생성
    fig = go.Figure(data=plotly_data)
    fig.update_layout(title='3D Segmentation Visualization', scene=dict(...), margin=dict(...)) # Layout 설정

    # HTML 파일 저장
    try:
        filename = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.html" # 확장자 html 로 변경
        save_dir_rel = os.path.join('plots', 'diagnosis', '3d_html') # 저장 경로 변경 (선택적)
        save_dir_abs = os.path.join(settings.MEDIA_ROOT, save_dir_rel)
        os.makedirs(save_dir_abs, exist_ok=True)
        save_path_abs = os.path.join(save_dir_abs, filename)

        # HTML 파일로 저장
        fig.write_html(save_path_abs, include_plotlyjs='cdn')
        print(f"Saved 3D visualization HTML to: {save_path_abs}")

        # MEDIA_ROOT 기준 상대 경로 반환
        return os.path.join(save_dir_rel, filename).replace("\\", "/")
    except Exception as e:
        print(f"Error saving 3D visualization HTML: {e}")
        traceback.print_exc()
        return None

# --- Gemini 해석 함수 (CT 용) ---
def get_gemini_interpretation_for_ct(classification_label, classification_prob, segmented_volume=None, scan_type="췌장"):
    """ CT 예측 결과를 바탕으로 Gemini AI를 사용하여 자연어 해석 생성 """
    # 이전 답변의 get_gemini_interpretation_for_ct 함수 내용 붙여넣기
    pass

# --- PDF 생성 유틸리티 ---
def generate_ct_diagnosis_pdf(result_id, context_dict={}):
    """ CT 진단 결과 PDF 생성하고 상대 경로 반환 (3D 이미지 제외) """
    try:
        template_path = 'diagnosis/pdf_report_template_ct.html'
        base_url = context_dict.get('request').build_absolute_uri('/') if context_dict.get('request') else settings.MEDIA_URL

        # 2D 이미지 URL 생성 로직은 유지
        context_dict['input_slice_plot_url'] = context_dict['request'].build_absolute_uri(settings.MEDIA_URL + context_dict['result'].input_image_slice_plot) if context_dict['result'].input_image_slice_plot else None
        context_dict['segmentation_map_plot_url'] = context_dict['request'].build_absolute_uri(settings.MEDIA_URL + context_dict['result'].segmentation_map_plot) if context_dict['result'].segmentation_map_plot else None
        context_dict['base_url'] = base_url

        # 3D 이미지 URL 생성 로직 제거
        # context_dict['visualization_3d_image_url'] = ... # 이 부분 제거

        font_config_path = os.path.join(settings.STATICFILES_DIRS[0], 'css', 'pdf_fonts.css') if settings.STATICFILES_DIRS else None
        stylesheets = [CSS(filename=font_config_path)] if font_config_path and os.path.exists(font_config_path) else []

        html_string = render_to_string(template_path, context_dict)
        html = HTML(string=html_string, base_url=base_url)

        pdf_filename = f"ct_report_{result_id}.pdf"
        pdf_dir_rel = os.path.join('pdfs', 'diagnosis')
        pdf_dir_abs = os.path.join(settings.MEDIA_ROOT, pdf_dir_rel)
        os.makedirs(pdf_dir_abs, exist_ok=True)
        pdf_full_path = os.path.join(pdf_dir_abs, pdf_filename)

        html.write_pdf(pdf_full_path, stylesheets=stylesheets)
        print(f"Generated PDF report (without 3D image): {pdf_full_path}")
        return os.path.join(pdf_dir_rel, pdf_filename).replace("\\", "/")

    except Exception as e:
        print(f"!!!!!!!! ERROR generating PDF for CT result {result_id} !!!!!!!!!")
        print(traceback.format_exc())
        return None