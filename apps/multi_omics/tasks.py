# apps/multi_omics/tasks.py
import time
import pandas as pd
from celery import shared_task
from django.utils import timezone
from django.conf import settings
from django.core.files.storage import default_storage
from .models import MultiOmicsRequest, MultiOmicsResult, MultiOmicsRequestStatus
from .utils import (
    load_columns, align_columns_and_fillna, predict_ensemble, # predict_cancer 대신 predict_ensemble 사용
    save_plot, save_gauge_plot, save_radar_plot,
    get_gemini_interpretation_for_multiomics, load_csv_from_media
)
import traceback
import os

@shared_task(bind=True, ignore_result=False) # ignore_result=False 로 결과 추적 가능하게 (선택 사항)
def run_multi_omics_prediction_task(self, request_id):
    """ Multi-omics 데이터 예측 Celery Task """
    print(f"[Task {self.request.id}] Processing MultiOmics Request {request_id}...")
    result_data_for_db = {} # 최종 결과를 담을 딕셔너리
    request = None # request 객체를 try 블록 밖에서도 접근 가능하게 초기화

    try:
        request = MultiOmicsRequest.objects.select_related('patient__user', 'requester').get(id=request_id)
        request.celery_task_id = self.request.id
        request.status = MultiOmicsRequestStatus.PROCESSING
        request.save(update_fields=['status', 'celery_task_id'])

        # --- 1. 데이터 로드 및 준비 ---
        print("   Loading and preparing data...")
        gene_data_raw = load_csv_from_media(request.gene_data_ref)
        protein_data_raw = load_csv_from_media(request.protein_data_ref)
        methylation_data_raw = load_csv_from_media(request.methylation_data_ref)
        cnv_data_raw = load_csv_from_media(request.cnv_data_ref)

        gene_cols = load_columns('GENE_COLUMNS_PATH')
        protein_cols = load_columns('PROTEIN_COLUMNS_PATH')
        methylation_cols = load_columns('METHYLATION_COLUMNS_PATH')
        cnv_cols = load_columns('CNV_COLUMNS_PATH')

        gene_aligned = align_columns_and_fillna(gene_data_raw, gene_cols)
        protein_aligned = align_columns_and_fillna(protein_data_raw, protein_cols)
        methylation_aligned = align_columns_and_fillna(methylation_data_raw, methylation_cols)
        cnv_aligned = align_columns_and_fillna(cnv_data_raw, cnv_cols)

        # --- 2. 예측 실행 ---
        print("   Running prediction ensemble...")
        ensemble_prob, individual_preds = predict_ensemble(
            gene_aligned, protein_aligned, methylation_aligned, cnv_aligned
        )
        result_data_for_db['ensemble_probability'] = ensemble_prob
        result_data_for_db['individual_probabilities'] = individual_preds

        # 임계값 (settings 또는 다른 곳에서 관리하는 것이 더 좋음)
        best_threshold = 0.5
        result_data_for_db['threshold_used'] = best_threshold
        ensemble_pred = None
        if ensemble_prob is not None:
             ensemble_pred = 1 if ensemble_prob >= best_threshold else 0
        result_data_for_db['ensemble_prediction'] = ensemble_pred

        # --- 3. 결과 시각화 ---
        print("   Generating plots...")
        # 저장 후 상대 경로(media 기준)를 DB에 저장
        result_data_for_db['gene_plot_path'] = save_plot(gene_aligned, "Gene Data Distribution", "gene")
        result_data_for_db['protein_plot_path'] = save_plot(protein_aligned, "Protein Data Distribution", "protein")
        result_data_for_db['methylation_plot_path'] = save_plot(methylation_aligned, "Methylation Data Distribution", "methylation")
        result_data_for_db['cnv_plot_path'] = save_plot(cnv_aligned, "CNV Data Distribution", "cnv")
        result_data_for_db['gauge_plot_path'] = save_gauge_plot(ensemble_prob)
        result_data_for_db['radar_plot_path'] = save_radar_plot(individual_preds)

        # --- 4. Gemini 해석 ---
        print("   Generating interpretation...")
        result_data_for_db['gemini_interpretation'] = get_gemini_interpretation_for_multiomics(
            ensemble_prob, individual_preds, request # request 객체 전달하여 컨텍스트 추가
        )

        # --- 5. 결과 저장 ---
        print("   Saving results to database...")
        result_data_for_db['error_message'] = "" # 성공 시 오류 메시지 초기화
        result_data_for_db['completion_timestamp'] = timezone.now()

        result_obj, created = MultiOmicsResult.objects.update_or_create(
            request=request,
            defaults=result_data_for_db # 준비된 딕셔너리 사용
        )
        request.status = MultiOmicsRequestStatus.COMPLETED
        request.save(update_fields=['status'])
        print(f"[Task {self.request.id}] MultiOmics Request {request_id} COMPLETED.")
        return {"status": "success", "result_id": result_obj.request_id} # 작업 성공 결과 반환

    except Exception as e:
        print(f"[Task {self.request.id}] FAILED processing MultiOmics Request {request_id}: {e}")
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(traceback.format_exc())
        # request 객체가 로드되었는지 확인 후 상태 업데이트 시도
        if request:
            try:
                request.status = MultiOmicsRequestStatus.FAILED
                request.save(update_fields=['status'])
                MultiOmicsResult.objects.update_or_create(
                    request=request,
                    defaults={
                        'error_message': error_msg,
                        'completion_timestamp': timezone.now(),
                        # 실패 시 다른 필드는 null 또는 기본값으로 남김
                        'ensemble_probability': None,
                        'ensemble_prediction': None,
                        'individual_probabilities': None,
                    }
                )
                print("   Saved FAILED status and error message to DB.")
            except Exception as update_err:
                print(f"   ERROR: Could not update multi-omics request/result status to FAILED: {update_err}")
        # Celery가 Task 실패를 인지하도록 예외 다시 발생
        self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': traceback.format_exc()})
        raise e # Re-raise the exception