# apps/diagnosis/tasks.py
import time
import os
from celery import shared_task
from django.utils import timezone
from django.conf import settings
from django.core.files.storage import default_storage
import torch
import traceback

from .models import DiagnosisRequest, DiagnosisResult, DiagnosisRequestStatus
from apps.patients.models import PatientProfile # PatientProfile 임포트
from .utils import (
    get_classification_transforms, get_segmentation_transforms,
    get_pancreas_classifier, get_pancreas_segmenter, run_inference,
    postprocess_pancreas_classification, postprocess_pancreas_segmentation,
    get_segmented_volume_voxels, save_plot_to_media, generate_nifti_slice_plot,
    generate_segmentation_overlay_plot, get_gemini_interpretation_for_ct,
    generate_ct_diagnosis_pdf, DEVICE,
    generate_3d_visualization_html,
)

@shared_task(bind=True, ignore_result=False)
def run_ct_diagnosis_task(self, request_id):
    """ CT 이미지 분류 및 분할 Celery Task """
    print(f"[Task {self.request.id}] Processing CT Diagnosis Request {request_id}...")
    request = None
    results_for_db = {} # 최종 결과를 담을 딕셔너리

    try:
        request = DiagnosisRequest.objects.select_related('patient__user', 'requester').get(id=request_id)
        request.celery_task_id = self.request.id
        request.status = DiagnosisRequestStatus.PROCESSING
        request.save(update_fields=['status', 'celery_task_id'])

        # --- 1. 입력 파일 경로 확인 ---
        nifti_relative_path = request.input_data_reference # DB에 저장된 상대 경로
        if not nifti_relative_path or not default_storage.exists(nifti_relative_path):
             raise FileNotFoundError(f"Input NIfTI file not found at: {nifti_relative_path}")
        nifti_absolute_path = default_storage.path(nifti_relative_path) # 절대 경로 얻기
        print(f"   Input NIfTI file: {nifti_absolute_path}")

        # --- 2. 분류 모델 처리 ---
        print("   Processing Classification...")
        classifier = get_pancreas_classifier() # 모델 로딩
        if not classifier: raise ValueError("Failed to load classifier model.")

        classify_transform = get_classification_transforms()
        # LoadImaged 는 파일 경로를 받으므로 절대 경로 사용
        classified_data_dict = classify_transform({'image': nifti_absolute_path})
        classified_input = classified_data_dict['image'].unsqueeze(0).to(DEVICE)

        classifier_output = run_inference(classifier, classified_input)
        if classifier_output is None: raise ValueError("Classifier inference failed.")

        class_prob, class_label = postprocess_pancreas_classification(classifier_output)
        if class_prob is None or class_label is None: raise ValueError("Classification postprocessing failed.")
        results_for_db['classification_probability'] = class_prob
        results_for_db['classification_prediction'] = class_label
        print(f"   Classification Result: Label={class_label}, Prob={class_prob:.4f}")

        # --- 3. 분할 모델 처리 ---
        print("   Processing Segmentation...")
        segmenter = get_pancreas_segmenter() # 모델 로딩
        if not segmenter: raise ValueError("Failed to load segmenter model.")

        segment_transform = get_segmentation_transforms()
        segmentation_data_dict = segment_transform({'image': nifti_absolute_path})
        segmentation_input = segmentation_data_dict['image'].unsqueeze(0).to(DEVICE)

        segmenter_output = run_inference(segmenter, segmentation_input)
        if segmenter_output is None: raise ValueError("Segmenter inference failed.")

        segmentation_mask_np = postprocess_pancreas_segmentation(segmenter_output)
        if segmentation_mask_np is None: raise ValueError("Segmentation postprocessing failed.")

        segmented_volume = get_segmented_volume_voxels(segmentation_mask_np, target_class=2) # 병변 클래스=2 가정
        results_for_db['segmentation_metrics'] = {'volume_voxels': segmented_volume}
        print(f"   Segmentation Result: Volume (voxels) = {segmented_volume}")
        
        # --- 3.5 3D 시각화 HTML 생성 ---
        print("   Generating 3D Visualization HTML...")
        vis_3d_html_rel_path = None 
        if segmentation_mask_np is not None:
            vis_3d_html_rel_path = generate_3d_visualization_html( 
                segmentation_mask_np,
                filename_prefix=f"ct_3d_vis_{request_id[:8]}"
            )
        results_for_db['visualization_3d_html_path'] = vis_3d_html_rel_path 

        # --- 4. 시각화 ---
        print("   Generating plots...")
        # utils의 save_plot_to_media 와 generate_*_plot 함수 사용
        input_plot_path = save_plot_to_media(generate_nifti_slice_plot, "ct_input_slice", nifti_absolute_path)
        seg_plot_path = save_plot_to_media(generate_segmentation_overlay_plot, "ct_seg_overlay", nifti_absolute_path, segmentation_mask_np)
        results_for_db['input_image_slice_plot'] = input_plot_path
        results_for_db['segmentation_map_plot'] = seg_plot_path

        # --- 5. Gemini 해석 ---
        print("   Generating interpretation...")
        results_for_db['gemini_interpretation'] = get_gemini_interpretation_for_ct(
            class_label, class_prob, segmented_volume, scan_type="췌장" # scan_type 은 request 모델에 저장하거나 여기서 지정
        )

        # --- 6. PDF 생성 ---
        print("   Generating PDF report...")
        # PDF 생성 함수는 context 를 필요로 함 (result 객체 대신 임시 context 생성)
        pdf_context = {
            'result': request, # 임시로 request 객체 전달 (내부 필드 접근용)
            'classification_prediction_display': '암 의심' if class_label == 1 else '정상',
            'classification_probability_percent': f"{class_prob*100:.1f}%",
            'segmentation_metrics': results_for_db['segmentation_metrics'],
            'gemini_interpretation': results_for_db['gemini_interpretation'],
            'input_image_slice_plot': results_for_db['input_image_slice_plot'], # 상대 경로
            'segmentation_map_plot': results_for_db['segmentation_map_plot'], # 상대 경로
            'request': request # utils 에서 절대 URL 생성 위해 필요
        }
        # 주의: generate_ct_diagnosis_pdf 함수가 context 를 유연하게 처리해야 함
        pdf_report_rel_path = generate_ct_diagnosis_pdf(request_id, pdf_context)
        results_for_db['pdf_report_path'] = pdf_report_rel_path


        # --- 7. 최종 결과 저장 ---
        print("   Saving final results to database...")
        results_for_db['error_message'] = ""
        results_for_db['completion_timestamp'] = timezone.now()

        # DB에 저장할 때 plot/pdf 경로는 MEDIA_ROOT 기준 상대 경로여야 함 (utils에서 그렇게 반환)
        # 필드 이름은 models.py 와 일치해야 함
        result_obj, created = DiagnosisResult.objects.update_or_create(
            request=request,
            defaults=results_for_db
        )

        request.status = DiagnosisRequestStatus.COMPLETED
        request.save(update_fields=['status'])
        print(f"[Task {self.request.id}] CT Diagnosis Request {request_id} COMPLETED.")
        return {"status": "success", "result_id": result_obj.request_id}

    except Exception as e:
        print(f"[Task {self.request.id}] FAILED processing CT Diagnosis Request {request_id}: {e}")
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(traceback.format_exc())
        if request:
            try:
                request.status = DiagnosisRequestStatus.FAILED
                request.save(update_fields=['status'])
                DiagnosisResult.objects.update_or_create(
                    request=request,
                    defaults={'error_message': error_msg, 'completion_timestamp': timezone.now()}
                )
                print("   Saved FAILED status and error message to DB.")
            except Exception as update_err:
                print(f"   ERROR: Could not update CT diagnosis request/result status to FAILED: {update_err}")
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': traceback.format_exc()})
        raise e # Celery 가 실패 인지하도록 예외 다시 발생

    finally:
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("   GPU memory cache cleared.")