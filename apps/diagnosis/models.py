import uuid
from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _
from apps.patients.models import PatientProfile # 외래키 위해 임포트

class DiagnosisRequestStatus(models.TextChoices): # 상태 코드 정의
    PENDING = 'PENDING', _('Pending')
    PROCESSING = 'PROCESSING', _('Processing')
    COMPLETED = 'COMPLETED', _('Completed')
    FAILED = 'FAILED', _('Failed')

class DiagnosisRequest(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    requester = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.PROTECT,
        related_name='ct_diagnosis_requests', # related_name 변경 (multi_omics 와 구분)
        limit_choices_to={'role': 'CLINICIAN'}, verbose_name=_("Requesting Clinician")
    )
    patient = models.ForeignKey(
        PatientProfile, on_delete=models.PROTECT,
        related_name='ct_diagnosis_requests', # related_name 변경
        verbose_name=_("Patient")
    )
    # NIfTI 파일 경로/키 저장 필드
    input_data_reference = models.CharField(
        _("Input Data Reference"), max_length=1024,
        help_text=_("Path or key for the input NIfTI data")
    )
    scan_type = models.CharField( # CT 앱에서는 scan_type 이 고정될 수 있으나, 확장성 위해 추가 가능
        _("Scan Type"), max_length=100, default='pancreas_ct', blank=True
    )
    status = models.CharField(
        _('Status'), max_length=20, choices=DiagnosisRequestStatus.choices,
        default=DiagnosisRequestStatus.PENDING, db_index=True
    )
    celery_task_id = models.CharField(
        _("Celery Task ID"), max_length=255, null=True, blank=True, db_index=True
    )
    request_timestamp = models.DateTimeField(auto_now_add=True, verbose_name=_("Request Time"))
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self): return f"CT Request {self.id} for {self.patient}"
    class Meta: verbose_name = _("CT Diagnosis Request"); verbose_name_plural = _("CT Diagnosis Requests"); ordering = ['-request_timestamp']

class DiagnosisResult(models.Model):
    request = models.OneToOneField(
        DiagnosisRequest, on_delete=models.CASCADE, primary_key=True,
        related_name='result', verbose_name=_("CT Diagnosis Request")
    )
    # 분류 결과
    classification_probability = models.FloatField(_("Classification Probability"), null=True, blank=True)
    classification_prediction = models.IntegerField(_("Classification Prediction"), null=True, blank=True) # 0 or 1
    # 분할 결과
    segmentation_metrics = models.JSONField(_("Segmentation Metrics"), null=True, blank=True) # e.g., {'volume_voxels': 123}
    # 시각화 결과 경로 (MEDIA_ROOT 기준 상대 경로)
    input_image_slice_plot = models.CharField(_("Input Slice Plot Path"), max_length=512, blank=True)
    segmentation_map_plot = models.CharField(_("Segmentation Plot Path"), max_length=512, blank=True)
    # PDF 보고서 경로
    pdf_report_path = models.CharField(_("PDF Report Path"), max_length=512, blank=True)
    # Gemini 해석
    gemini_interpretation = models.TextField(_("Gemini Interpretation"), blank=True)
    # 기타 (결과 요약 등 필요시 추가)
    result_summary = models.CharField(_("Result Summary"), max_length=100, blank=True) # 예: "암 의심", "정상"
    # 오류
    error_message = models.TextField(_("Error Message"), blank=True)
    completion_timestamp = models.DateTimeField(_("Completion Time"), null=True, blank=True)
    # PDF
    pdf_report_path = models.CharField(_("PDF Report Path"), max_length=512, blank=True)
    visualization_3d_html_path = models.CharField(
        _("3D Visualization HTML Path"), max_length=512, blank=True, null=True)
    gemini_interpretation = models.TextField(_("Gemini Interpretation"), blank=True)

    def __str__(self): return f"Result for CT Request {self.request_id}"
    class Meta: verbose_name = _("CT Diagnosis Result"); verbose_name_plural = _("CT Diagnosis Results")

    # 결과 요약 자동 설정 (선택적)
    def save(self, *args, **kwargs):
        if self.classification_prediction == 1: self.result_summary = "암 의심"
        elif self.classification_prediction == 0: self.result_summary = "정상"
        else: self.result_summary = "판독 불가" if self.error_message else ""
        super().save(*args, **kwargs)