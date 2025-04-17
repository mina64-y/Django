# apps/diagnosis/admin.py
from django.contrib import admin
from .models import DiagnosisRequest, DiagnosisResult

class DiagnosisResultInline(admin.StackedInline):
    model = DiagnosisResult
    can_delete = False
    verbose_name_plural = 'Diagnosis Result'
    # 실제 DiagnosisResult 모델에 있는 필드 이름으로 수정/추가
    readonly_fields = (
        'completion_timestamp',
        'result_summary', # models.py 에 CharField로 정의됨
        'classification_probability', # confidence_score 대신 사용
        'classification_prediction', # 추가
        'segmentation_metrics', # additional_data 대신 사용
        'input_image_slice_plot', # 추가
        'segmentation_map_plot', # 추가
        'visualization_3d_html_path', # 또는 visualization_3d_image_path (모델 필드명 확인)
        'pdf_report_path',
        'gemini_interpretation',
        'error_message',
    )
    # 필드가 너무 많으면 fieldsets 를 사용하여 그룹화할 수도 있습니다.
    # fieldsets = (
    #     (None, {'fields': ('completion_timestamp', 'result_summary', 'error_message')}),
    #     ('Classification', {'fields': ('classification_probability', 'classification_prediction')}),
    #     ('Segmentation', {'fields': ('segmentation_metrics',)}),
    #     ('Files', {'fields': ('input_image_slice_plot', 'segmentation_map_plot', 'visualization_3d_html_path', 'pdf_report_path')}),
    #     ('Interpretation', {'fields': ('gemini_interpretation',)}),
    # )
    extra = 0 # 새 결과 인라인 추가 방지

@admin.register(DiagnosisRequest)
class DiagnosisRequestAdmin(admin.ModelAdmin):
    list_display = ('id', 'requester', 'patient', 'status', 'request_timestamp', 'celery_task_id')
    list_filter = ('status', 'request_timestamp', 'requester')
    search_fields = ('id', 'requester__email', 'patient__user__email')
    readonly_fields = ('id', 'requester', 'patient', 'input_data_reference', 'scan_type', # scan_type 추가
                       'celery_task_id', 'request_timestamp', 'updated_at')
    inlines = [DiagnosisResultInline] # 결과 인라인 표시

# Result 모델은 Request 를 통해 관리되므로 별도 등록 안 함
# @admin.register(DiagnosisResult)
# class DiagnosisResultAdmin(admin.ModelAdmin):
#     list_display = ('request_id', 'result_summary', 'completion_timestamp')
#     search_fields = ('request__id',)