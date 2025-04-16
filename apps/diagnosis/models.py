import uuid
from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _

class DiagnosisRequestStatus(models.TextChoices):
    PENDING = 'PENDING', _('Pending')
    PROCESSING = 'PROCESSING', _('Processing')
    COMPLETED = 'COMPLETED', _('Completed')
    FAILED = 'FAILED', _('Failed')

class DiagnosisRequest(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    requester = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.PROTECT, # Prevent deletion of user if requests exist
        related_name='diagnosis_requests',
        limit_choices_to={'role': 'CLINICIAN'}, # Only Clinicians can request
        verbose_name=_("Requesting Clinician")
    )
    patient = models.ForeignKey(
        'patients.PatientProfile', # Use string notation to avoid circular import
        on_delete=models.PROTECT,
        related_name='diagnosis_requests',
        verbose_name=_("Patient")
    )
    # Store reference to uploaded data (e.g., file path in cloud storage)
    # Option 1: Store path directly (simpler if using pre-signed URLs)
    input_data_reference = models.CharField(
        _("Input Data Reference"),
        max_length=1024,
        help_text=_("Path or identifier for the input data (e.g., S3 key)")
    )
    # Option 2: Link to a separate File model (more structured)
    # uploaded_file = models.ForeignKey('files.UploadedFile', ...)

    status = models.CharField(
        _('Status'),
        max_length=20,
        choices=DiagnosisRequestStatus.choices,
        default=DiagnosisRequestStatus.PENDING,
        db_index=True
    )
    celery_task_id = models.CharField(
        _("Celery Task ID"), max_length=255, null=True, blank=True, db_index=True
    )
    request_timestamp = models.DateTimeField(auto_now_add=True, verbose_name=_("Request Time"))
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Request {self.id} for {self.patient}"

    class Meta:
        verbose_name = _("Diagnosis Request")
        verbose_name_plural = _("Diagnosis Requests")
        ordering = ['-request_timestamp']

class DiagnosisResult(models.Model):
    request = models.OneToOneField(
        DiagnosisRequest,
        on_delete=models.CASCADE, # If request is deleted, result is too
        primary_key=True,
        related_name='result',
        verbose_name=_("Diagnosis Request")
    )
    result_summary = models.TextField(_("Result Summary"), blank=True) # e.g., "Positive", "Negative", "Indeterminate"
    confidence_score = models.FloatField(_("Confidence Score"), null=True, blank=True)
    # Store path relative to MEDIA_ROOT
    pdf_report_path = models.CharField(_("PDF Report Path"), max_length=512, blank=True)
    # Store Gemini's interpretation
    gemini_interpretation = models.TextField(_("Gemini Interpretation"), blank=True)
    # Store any structured data from the model if needed (e.g., segmentation metrics)
    additional_data = models.JSONField(_("Additional Data"), null=True, blank=True)
    error_message = models.TextField(_("Error Message"), blank=True) # Store errors from Celery task
    completion_timestamp = models.DateTimeField(_("Completion Time"), null=True, blank=True)

    def __str__(self):
        return f"Result for Request {self.request_id}"

    class Meta:
        verbose_name = _("Diagnosis Result")
        verbose_name_plural = _("Diagnosis Results")