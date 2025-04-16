# apps/diagnosis/admin.py
from django.contrib import admin
from .models import DiagnosisRequest, DiagnosisResult

class DiagnosisResultInline(admin.StackedInline): # 요청 상세 페이지에서 결과를 같이 보기 위함
    model = DiagnosisResult
    can_delete = False
    verbose_name_plural = 'Diagnosis Result'
    readonly_fields = ('completion_timestamp', 'result_summary', 'confidence_score', 'pdf_report_path', 'gemini_interpretation', 'additional_data', 'error_message')

@admin.register(DiagnosisRequest)
class DiagnosisRequestAdmin(admin.ModelAdmin):
    list_display = ('id', 'requester', 'patient', 'status', 'request_timestamp', 'celery_task_id')
    list_filter = ('status', 'request_timestamp', 'requester')
    search_fields = ('id', 'requester__email', 'patient__user__email')
    readonly_fields = ('id', 'requester', 'patient', 'input_data_reference', 'celery_task_id', 'request_timestamp', 'updated_at')
    inlines = [DiagnosisResultInline] # 결과 인라인 표시

# Result 모델은 Request 를 통해 관리되므로 별도 등록 안 함 (필요시 등록 가능)
# @admin.register(DiagnosisResult)
# class DiagnosisResultAdmin(admin.ModelAdmin):
#     list_display = ('request_id', 'result_summary', 'completion_timestamp')
#     search_fields = ('request__id',)