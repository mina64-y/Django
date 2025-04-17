# 참고: 위 코드에서 폰트 설정(font_config_path 또는 font_config) 부분은 
# 실제 프로젝트의 정적 파일 설정 및 폰트 파일 위치에 맞게 조정해야 합니다.
# /static/css/pdf_fonts.css 파일에 @font-face 와 
# body { font-family: ... } 를 정의하는 것이 좋습니다.

# apps/multi_omics/utils.py
import joblib
import os
import uuid
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # GUI 없는 환경용 백엔드 설정
import matplotlib.pyplot as plt
from django.conf import settings
from django.core.files.storage import default_storage # 파일 시스템 접근용
import traceback
import google.generativeai as genai # Gemini 라이브러리 import
from django.template.loader import render_to_string

# --- Gemini 설정 ---
GEMINI_API_KEY = settings.GEMINI_API_KEY
GEMINI_CONFIGURED = False
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # 모델 초기화 (애플리케이션 시작 시 또는 첫 호출 시)
        gemini_model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest") # 또는 다른 모델
        GEMINI_CONFIGURED = True
        print("Gemini AI configured successfully.")
    except Exception as e:
        print(f"!!!!!!!! ERROR configuring Gemini AI !!!!!!!!!\n{e}")
        traceback.print_exc()
else:
    print("WARN: GEMINI_API_KEY not found in settings. Gemini interpretation disabled.")


# --- 모델 및 컬럼 로딩 유틸리티 ---
def load_joblib_model(model_path_setting_name):
    """ settings.py 에 정의된 경로로 joblib 모델 로드 """
    model_path = getattr(settings, model_path_setting_name, None)
    if model_path and os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            print(f"Loaded model from: {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            traceback.print_exc()
    else:
        print(f"Model path not found or invalid in settings: {model_path_setting_name} -> {model_path}")
    return None

def load_columns(columns_path_setting_name):
    """ settings.py 에 정의된 경로로 컬럼 리스트(.pkl) 로드 """
    columns_path = getattr(settings, columns_path_setting_name, None)
    if columns_path and os.path.exists(columns_path):
         try:
             columns = joblib.load(columns_path)
             print(f"Loaded columns from: {columns_path}")
             return columns
         except Exception as e:
             print(f"Error loading columns {columns_path}: {e}")
             traceback.print_exc()
    else:
        print(f"Columns path not found or invalid in settings: {columns_path_setting_name} -> {columns_path}")
    return None

# --- 데이터 처리 유틸리티 ---
def load_csv_from_media(relative_path):
    """ MEDIA_ROOT 기준 상대 경로로 CSV 파일 로드 """
    if not relative_path: return None
    try:
        if default_storage.exists(relative_path):
            with default_storage.open(relative_path, 'r') as f:
                # 인코딩 문제 발생 시 encoding='utf-8' 또는 'cp949' 추가 고려
                df = pd.read_csv(f)
                print(f"Loaded CSV data from: {relative_path}, shape: {df.shape}")
                return df
        else:
            print(f"Data file not found at media path: {relative_path}")
            return None
    except Exception as e:
        print(f"Error loading CSV data from {relative_path}: {e}")
        traceback.print_exc()
        return None

# (이전 답변의 align_columns_and_fillna 함수 내용과 동일하게 유지)
def align_columns_and_fillna(data_df, required_cols):
     """ 데이터프레임 컬럼 맞추고 NaN/None 을 0으로 채우기 """
     if data_df is None or data_df.empty or not required_cols:
         return pd.DataFrame(columns=required_cols if required_cols else [])
     aligned_df = data_df.reindex(columns=required_cols, fill_value=0)
     aligned_df.fillna(0, inplace=True)
     return aligned_df


# --- 예측 로직 ---
def predict_ensemble(gene_aligned, protein_aligned, methylation_aligned, cnv_aligned):
     """ 정렬된 데이터로 앙상블 예측 수행 (predict_cancer 역할)"""
     # 모델 로딩 (Task 실행 시 호출되므로 매번 로드)
     loaded_models = {
         'gene': load_joblib_model('GENE_MODEL_PATH'),
         'protein': load_joblib_model('PROTEIN_MODEL_PATH'),
         'methylation': load_joblib_model('METHYLATION_MODEL_PATH'),
         'cnv': load_joblib_model('CNV_MODEL_PATH')
     }
     aligned_data = {
         'gene': gene_aligned,
         'protein': protein_aligned,
         'methylation': methylation_aligned,
         'cnv': cnv_aligned
     }

     individual_probs = {} # 개별 모델 예측 확률 저장 (클래스 1 확률)
     available_models = [] # 예측에 사용된 모델 이름

     def is_valid_dataframe(df):
         return isinstance(df, pd.DataFrame) and not df.empty

     # 각 모델별 예측
     for name, model in loaded_models.items():
         data = aligned_data.get(name)
         if model and is_valid_dataframe(data):
             try:
                 # predict_proba 결과는 보통 [[class0_prob, class1_prob], ...] 형태
                 # 입력 데이터가 여러 행일 수 있으므로 첫 번째 행만 사용 (API 설계에 따라 변경 필요)
                 proba = model.predict_proba(data)[:, 1]
                 individual_probs[name] = proba[0] if len(proba) > 0 else None # 첫번째 샘플 확률 저장
                 if individual_probs[name] is not None:
                    available_models.append(name)
             except AttributeError: # predict_proba 없는 모델 (SVM 등)
                  try:
                      # decision_function 결과 -> 확률 변환 (Sigmoid)
                      decision_values = model.decision_function(data)
                      proba = 1 / (1 + np.exp(-decision_values))
                      individual_probs[name] = proba[0] if len(proba) > 0 else None
                      if individual_probs[name] is not None:
                          available_models.append(name)
                  except Exception as e_dec:
                      print(f"Error getting decision_function/proba for {name}: {e_dec}")
                      individual_probs[name] = None
             except Exception as e_pred:
                 print(f"Error predicting with {name} model: {e_pred}")
                 individual_probs[name] = None
         else:
             individual_probs[name] = None # 모델 또는 데이터 없음

     # 앙상블: 가중 평균 (F1 점수 기반 - 하드코딩된 값 사용)
     f1_scores = {'gene': 0.95, 'protein': 0.936, 'methylation': 0.9855, 'cnv': 0.9865}
     total_f1 = sum(f1_scores[model] for model in available_models if model in f1_scores)
     ensemble_prob = None

     if total_f1 > 0 and available_models:
         weighted_sum = 0
         for model_name in available_models:
             if model_name in f1_scores and individual_probs.get(model_name) is not None:
                 weighted_sum += individual_probs[model_name] * (f1_scores[model_name] / total_f1)
         ensemble_prob = weighted_sum
     elif available_models: # 가중치 없는 경우 단순 평균 (또는 다른 전략)
         valid_probs = [individual_probs[m] for m in available_models if individual_probs.get(m) is not None]
         if valid_probs:
            ensemble_prob = sum(valid_probs) / len(valid_probs)

     print(f"Available models for ensemble: {available_models}")
     print(f"Individual Probabilities: {individual_probs}")
     print(f"Ensemble Probability: {ensemble_prob}")

     return ensemble_prob, individual_probs # 앙상블 확률, 개별 확률 딕셔너리

# --- 플롯 생성 유틸리티 ---
# (이전 답변의 save_plot, save_gauge_plot, save_radar_plot 함수 내용과 동일하게 유지)
def save_plot(data, title, prefix, max_cols=50):
    """ 데이터 분포 플롯 저장 (예시) """
    if data is None or data.empty: return None
    try:
        plot_data = data.iloc[0] # 첫 번째 행 데이터만 사용 (가정)
        if len(plot_data) > max_cols:
             plot_data = plot_data.nlargest(max_cols // 2).append(plot_data.nsmallest(max_cols // 2))

        fig, ax = plt.subplots(figsize=(10, 4))
        plot_data.plot(kind='bar', ax=ax)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        plt.tight_layout()

        filename = f"{prefix}_dist_{uuid.uuid4().hex[:8]}.png"
        # default_storage 를 사용하여 media 경로에 저장
        save_path_rel = os.path.join('plots', filename) # media 아래 plots 폴더에 저장 가정
        os.makedirs(os.path.join(settings.MEDIA_ROOT, 'plots'), exist_ok=True)
        save_path_abs = os.path.join(settings.MEDIA_ROOT, save_path_rel)

        plt.savefig(save_path_abs)
        plt.close(fig)
        print(f"Saved plot: {save_path_abs}")
        # 저장된 파일의 상대 경로 반환 (MEDIA_ROOT 기준)
        return save_path_rel
    except Exception as e:
        print(f"Error saving plot '{title}': {e}")
        traceback.print_exc()
        if 'fig' in locals(): plt.close(fig)
        return None

def save_gauge_plot(prob, prefix="gauge"):
    """ 게이지 차트 저장 """
    if prob is None: return None
    try:
        fig, ax = plt.subplots(figsize=(6, 1.5))
        val = float(prob)
        ax.barh(0, 1, height=0.5, color='lightgray', edgecolor='darkgray')
        bar_color = 'mediumslateblue' if val < 0.5 else 'lightcoral'
        ax.barh(0, val, height=0.5, color=bar_color, edgecolor='black')
        ax.set_xlim(0, 1); ax.set_ylim(-0.5, 0.5)
        ax.set_title(f"Ensemble Prediction Probability: {val:.1%}", fontsize=12, weight='bold')
        ax.set_yticks([]); ax.set_xlabel("Probability (0 to 1)")
        ax.text(val + 0.02 if val < 0.9 else val - 0.1, 0, f"{val:.1%}", va='center', ha='left' if val < 0.9 else 'right', fontsize=10, weight='bold')

        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
        save_path_rel = os.path.join('plots', filename)
        os.makedirs(os.path.join(settings.MEDIA_ROOT, 'plots'), exist_ok=True)
        save_path_abs = os.path.join(settings.MEDIA_ROOT, save_path_rel)

        plt.tight_layout(pad=0.5)
        plt.savefig(save_path_abs, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved gauge plot: {save_path_abs}")
        return save_path_rel # 상대 경로 반환
    except Exception as e:
        print(f"Error saving gauge plot: {e}")
        traceback.print_exc()
        if 'fig' in locals(): plt.close(fig)
        return None

def save_radar_plot(pred_dict, prefix="radar"):
    """ 레이더 차트 저장 """
    if not pred_dict: return None
    valid_dict = {k: (v if v is not None else 0) for k, v in pred_dict.items() if k in ['gene', 'protein', 'methylation', 'cnv']} # 순서 고정 및 유효 키 필터링
    labels = list(valid_dict.keys())
    if not labels: return None

    values = [valid_dict[key] for key in labels]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]; angles += angles[:1]

    try:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, color='teal', linewidth=2, linestyle='solid')
        ax.fill(angles, values, color='teal', alpha=0.3)
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels)
        ax.set_yticks(np.linspace(0, 1, 5)); ax.set_yticklabels([f"{i*25}%" for i in range(5)])
        ax.set_ylim(0, 1); ax.set_title("Individual Model Probabilities", size=14, y=1.1)

        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
        save_path_rel = os.path.join('plots', filename)
        os.makedirs(os.path.join(settings.MEDIA_ROOT, 'plots'), exist_ok=True)
        save_path_abs = os.path.join(settings.MEDIA_ROOT, save_path_rel)

        plt.savefig(save_path_abs, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved radar plot: {save_path_abs}")
        return save_path_rel # 상대 경로 반환
    except Exception as e:
        print(f"Error saving radar plot: {e}")
        traceback.print_exc()
        if 'fig' in locals(): plt.close(fig)
        return None


# --- Gemini 해석 유틸리티 ---
def get_gemini_interpretation_for_multiomics(ensemble_prob, individual_preds, request_info=None):
    if not GEMINI_CONFIGURED or gemini_model is None:
        return "Gemini AI 해석 기능이 설정되지 않았거나 설정에 실패했습니다."

    if ensemble_prob is None and not any(v is not None for v in individual_preds.values()):
        return "모델 예측 결과가 없어 해석을 생성할 수 없습니다."

    try:
        prob_summary = f"전체 앙상블 예측 확률: {ensemble_prob:.1%}\n" if ensemble_prob is not None else "전체 앙상블 예측을 수행할 수 없었습니다.\n"
        individual_summary = ""
        # 표시 순서 지정 (선택적)
        ordered_keys = ['gene', 'protein', 'methylation', 'cnv']
        for key in ordered_keys:
            prob = individual_preds.get(key)
            if prob is not None:
                 individual_summary += f"- {key.capitalize()} 기반 예측 확률: {prob:.1%}\n"
            # else: # 예측 불가 표시는 제외하거나 필요시 추가
            #     individual_summary += f"- {key.capitalize()} 데이터 예측 불가\n"

        context_info = f"환자 ID: {request_info.patient.user.id}\n요청 시간: {request_info.request_timestamp.strftime('%Y-%m-%d %H:%M')}\n" if request_info else ""

        prompt = f"""
        다음은 특정 환자에 대한 다중 오믹스(유전자, 단백질, 메틸화, CNV) 기반 췌장암 예측 결과입니다.

        {context_info}
        [예측 결과 요약]
        {prob_summary}
        [개별 데이터 기반 예측]
        {individual_summary}
        ---
        위 예측 결과를 바탕으로, 일반 사용자가 이해하기 쉽게 **친절하고 명확한 언어**로 결과를 요약하고 설명해주세요.
        각 데이터 유형별 결과가 예측에 어떻게 기여했는지 간단히 언급하고, 최종 앙상블 확률의 의미를 설명해주세요.
        확률 수치를 포함하되, 확진이 아님을 명확히 하고 추가적인 검사나 전문가 상담의 필요성을 부드럽게 언급해주세요.
        결과가 좋지 않더라도 희망을 잃지 않도록 긍정적인 어조를 유지해주세요.
        **매우 중요**: 이 AI 예측은 참고 자료일 뿐이며, **절대 최종 진단이 아님**을 명확히 강조하고, **반드시 의사와 상담**해야 한다는 강력한 경고 문구를 포함해주세요.
        """

        # 안전 설정
        safety_settings = [
             {"category": "HARM_CATEGORY_MEDICAL", "threshold": "BLOCK_ONLY_HIGH"}, # 의료 정보 허용 수준 조절 필요
             {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
             {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
             {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
             {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
         ]
        response = gemini_model.generate_content(prompt, safety_settings=safety_settings)

        # 응답 처리
        if not response.parts:
             block_reason = "알 수 없음"
             try:
                 if response.prompt_feedback and response.prompt_feedback.block_reason:
                     block_reason = response.prompt_feedback.block_reason.name
             except AttributeError: pass
             print(f"WARN: Gemini 요청이 차단되었습니다. 이유: {block_reason}")
             return f"AI 해석 생성 중 콘텐츠 안전 문제로 결과 표시 불가 (이유: {block_reason}). 관리자 문의."

        interpretation = response.text.strip()
        print("Generated Gemini interpretation.")
        return interpretation

    except Exception as e:
        print(f"!!!!!!!! ERROR calling Gemini API !!!!!!!!!")
        print(traceback.format_exc())
        return f"AI 해석 생성 중 오류 발생 ({type(e).__name__}). 관리자 문의."
    
# --- Multi-omics PDF 생성 유틸리티 ---
def generate_multi_omics_pdf(result_obj):
    """ MultiOmicsResult 객체를 받아 PDF 생성 후 상대 경로 반환 """
    if not isinstance(result_obj, MultiOmicsResult):
        print("ERROR: Invalid MultiOmicsResult object passed to PDF generator.")
        return None

    try:
        request_id = result_obj.request_id
        template_path = 'multi_omics/pdf_report_template_omics.html'

        # 이미지 파일의 절대 URL 생성 (MEDIA_URL 사용)
        def get_absolute_media_url(relative_path):
            if relative_path:
                # 운영 환경에서는 settings.SITE_URL 같은 것을 정의하여 사용하거나,
                # request 객체가 있다면 request.build_absolute_uri 사용
                # 여기서는 간단히 MEDIA_URL 만 사용 (로컬 개발 환경 가정)
                # 실제 배포 시에는 전체 URL 생성이 필요할 수 있음
                if settings.MEDIA_URL.startswith('http'):
                     return f"{settings.MEDIA_URL.rstrip('/')}/{relative_path.lstrip('/')}"
                else: # 상대 경로면 그대로 사용 (WeasyPrint base_url 에서 처리)
                     return f"{settings.MEDIA_URL}{relative_path}"
            return None

        context = {
            'result': result_obj,
            # 플롯 이미지 URL 전달
            'gene_plot_url': get_absolute_media_url(result_obj.gene_plot_path),
            'protein_plot_url': get_absolute_media_url(result_obj.protein_plot_path),
            'methylation_plot_url': get_absolute_media_url(result_obj.methylation_plot_path),
            'cnv_plot_url': get_absolute_media_url(result_obj.cnv_plot_path),
            'gauge_plot_url': get_absolute_media_url(result_obj.gauge_plot_path),
            'radar_plot_url': get_absolute_media_url(result_obj.radar_plot_path),
            # base_url: 템플릿 내 상대 경로 리소스(예: CSS, 로고 이미지) 해석 기준
            'base_url': settings.MEDIA_URL # 또는 settings.STATIC_URL 등 기준 경로
        }

        # 한글 폰트 설정 (CT PDF 와 동일하게)
        # font_config = CSS(string='@font-face { font-family: "Malgun Gothic"; src: url(/static/fonts/malgun.ttf); } body { font-family: "Malgun Gothic"; }')
        font_config_path = os.path.join(settings.STATICFILES_DIRS[0], 'css', 'pdf_fonts.css') if settings.STATICFILES_DIRS else None
        stylesheets = [CSS(filename=font_config_path)] if font_config_path and os.path.exists(font_config_path) else []
        # 또는 CSS 객체 직접 생성
        # font_config = CSS(string='@font-face { ... } body { font-family: ...;}')
        # stylesheets=[font_config]


        html_string = render_to_string(template_path, context)
        html = HTML(string=html_string, base_url=context.get('base_url', settings.MEDIA_URL)) # 이미지 경로 해석 기준

        # 파일명 및 저장 경로 설정 (media/pdfs/multi_omics/)
        pdf_filename = f"multi_omics_report_{request_id}.pdf"
        pdf_dir_rel = os.path.join('pdfs', 'multi_omics')
        pdf_dir_abs = os.path.join(settings.MEDIA_ROOT, pdf_dir_rel)
        os.makedirs(pdf_dir_abs, exist_ok=True)
        pdf_full_path = os.path.join(pdf_dir_abs, pdf_filename)

        html.write_pdf(pdf_full_path, stylesheets=stylesheets) # 폰트 적용
        print(f"Generated Multi-omics PDF report: {pdf_full_path}")

        # MEDIA_ROOT 기준 상대 경로 반환
        return os.path.join(pdf_dir_rel, pdf_filename).replace("\\", "/")

    except Exception as e:
        print(f"!!!!!!!! ERROR generating PDF for MultiOmics request {result_obj.request_id} !!!!!!!!!")
        print(traceback.format_exc())
        return None