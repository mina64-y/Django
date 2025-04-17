# apps/multi_omics/models.py
import uuid
from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _

# 진단 앱의 상태 코드를 재사용하거나 별도로 정의할 수 있습니다.
# from apps.diagnosis.models import DiagnosisRequestStatus
class MultiOmicsRequestStatus(models.TextChoices):
    PENDING = 'PENDING', _('Pending')
    PROCESSING = 'PROCESSING', _('Processing')
    COMPLETED = 'COMPLETED', _('Completed')
    FAILED = 'FAILED', _('Failed')

class MultiOmicsRequest(models.Model):
    """ Multi-omics (gene, protein, etc.) 예측 요청 """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    requester = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT,
        related_name='multi_omics_requests',
        limit_choices_to={'role': 'CLINICIAN'},
        verbose_name=_("Requesting Clinician")
    )
    patient = models.ForeignKey(
        'patients.PatientProfile',
        on_delete=models.PROTECT,
        related_name='multi_omics_requests',
        verbose_name=_("Patient")
    )
    # 입력 데이터 참조 방식 정의 필요
    # 예시 1: 각 파일 경로를 저장 (클라이언트가 업로드 후 경로 전달 가정)
    gene_data_ref = models.CharField(_("Gene Data Ref"), max_length=1024, blank=True)
    protein_data_ref = models.CharField(_("Protein Data Ref"), max_length=1024, blank=True)
    methylation_data_ref = models.CharField(_("Methylation Data Ref"), max_length=1024, blank=True)
    cnv_data_ref = models.CharField(_("CNV Data Ref"), max_length=1024, blank=True)
    # 예시 2: JSON 필드에 파일 정보 저장
    # input_references = models.JSONField(_("Input References"), blank=True, null=True)

    status = models.CharField(
        _('Status'),
        max_length=20,
        choices=MultiOmicsRequestStatus.choices,
        default=MultiOmicsRequestStatus.PENDING,
        db_index=True
    )
    celery_task_id = models.CharField(
        _("Celery Task ID"), max_length=255, null=True, blank=True, db_index=True
    )
    request_timestamp = models.DateTimeField(auto_now_add=True, verbose_name=_("Request Time"))
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"MultiOmics Request {self.id} for {self.patient}"

    class Meta:
        verbose_name = _("Multi-omics Request")
        verbose_name_plural = _("Multi-omics Requests")
        ordering = ['-request_timestamp']

class MultiOmicsResult(models.Model):
    """ Multi-omics 예측 결과 """
    request = models.OneToOneField(
        MultiOmicsRequest,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name='result',
        verbose_name=_("Multi-omics Request")
    )
    # 예측 결과
    ensemble_probability = models.FloatField(_("Ensemble Probability"), null=True, blank=True)
    ensemble_prediction = models.IntegerField(_("Ensemble Prediction"), null=True, blank=True) # 예: 0 or 1
    # 개별 모델 결과 (JSON으로 저장하는 것이 유연할 수 있음)
    individual_probabilities = models.JSONField(_("Individual Probabilities"), null=True, blank=True)
    # 사용된 임계값 (참고용)
    threshold_used = models.FloatField(_("Threshold Used"), null=True, blank=True)

    # 생성된 플롯 경로
    gene_plot_path = models.CharField(_("Gene Plot Path"), max_length=512, blank=True, null=True) # null=True 추가
    protein_plot_path = models.CharField(_("Protein Plot Path"), max_length=512, blank=True, null=True)
    methylation_plot_path = models.CharField(_("Methylation Plot Path"), max_length=512, blank=True, null=True)
    cnv_plot_path = models.CharField(_("CNV Plot Path"), max_length=512, blank=True, null=True)
    gauge_plot_path = models.CharField(_("Gauge Plot Path"), max_length=512, blank=True, null=True)
    radar_plot_path = models.CharField(_("Radar Plot Path"), max_length=512, blank=True, null=True)

    # --- PDF 보고서 경로 필드 추가 ---
    pdf_report_path = models.CharField(
        _("PDF Report Path"), max_length=512, blank=True, null=True # null=True 추가
    )

    # Gemini 해석
    gemini_interpretation = models.TextField(_("Gemini Interpretation"), blank=True)
    error_message = models.TextField(_("Error Message"), blank=True)
    completion_timestamp = models.DateTimeField(_("Completion Time"), null=True, blank=True)
    
    # 오류 메시지
    error_message = models.TextField(_("Error Message"), blank=True)
    completion_timestamp = models.DateTimeField(_("Completion Time"), null=True, blank=True)

    def __str__(self):
        return f"Result for MultiOmics Request {self.request_id}"

    class Meta:
        verbose_name = _("Multi-omics Result")
        verbose_name_plural = _("Multi-omics Results")