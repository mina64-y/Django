# apps/multi_omics/admin.py
from django.contrib import admin
from .models import MultiOmicsRequest, MultiOmicsResult

class MultiOmicsResultInline(admin.StackedInline):
    model = MultiOmicsResult
    can_delete = False
    verbose_name_plural = 'Multi-omics Result'
    # 결과 필드들을 읽기 전용으로 표시
    readonly_fields = ('ensemble_probability', 'ensemble_prediction', 'individual_probabilities',
                       'threshold_used', 'gene_plot_path', 'protein_plot_path',
                       'methylation_plot_path', 'cnv_plot_path', 'gauge_plot_path',
                       'radar_plot_path', 'gemini_interpretation', 'error_message',
                       'completion_timestamp')
    extra = 0 # 새 결과 바로 추가 방지

@admin.register(MultiOmicsRequest)
class MultiOmicsRequestAdmin(admin.ModelAdmin):
    list_display = ('id', 'requester', 'patient', 'status', 'request_timestamp', 'celery_task_id')
    list_filter = ('status', 'request_timestamp', 'requester')
    search_fields = ('id', 'requester__email', 'patient__user__email')
    readonly_fields = ('id', 'requester', 'patient', 'celery_task_id',
                       'request_timestamp', 'updated_at',
                       'gene_data_ref', 'protein_data_ref', 'methylation_data_ref', 'cnv_data_ref') # 입력 참조는 수정 불가
    inlines = [MultiOmicsResultInline]