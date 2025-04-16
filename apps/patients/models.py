from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _
from django.core.exceptions import ValidationError

# Link Patient Profile to the User model
class PatientProfile(models.Model):
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name='patient_profile',
        limit_choices_to={'role': 'PATIENT'} # Ensure only users with PATIENT role can have this profile
    )
    date_of_birth = models.DateField(_('Date of Birth'), null=True, blank=True)
    # Add other demographic fields as needed (address, phone, etc.)
    # PIPA: Consider if fields are sensitive, need specific consent tracking

    # Consent fields (Example - needs robust implementation based on PIPA)
    # consent_data_processing = models.BooleanField(default=False, verbose_name=_("Data Processing Consent"))
    # consent_medication_tracking = models.BooleanField(default=False, verbose_name=_("Medication Tracking Consent"))

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Profile for {self.user.email}"

    class Meta:
        verbose_name = _("Patient Profile")
        verbose_name_plural = _("Patient Profiles")


class Medication(models.Model):
    """ Represents medication taken by a patient """
    patient_profile = models.ForeignKey(
        PatientProfile,
        on_delete=models.CASCADE,
        related_name='medications',
        verbose_name=_("Patient Profile")
    )
    name = models.CharField(_('Medication Name'), max_length=200)
    dosage = models.CharField(_('Dosage'), max_length=100, blank=True) # e.g., "100mg", "1 tablet"
    frequency = models.CharField(_('Frequency'), max_length=100, blank=True) # e.g., "Once daily", "Twice daily"
    start_date = models.DateField(_('Start Date'))
    end_date = models.DateField(_('End Date'), null=True, blank=True)
    notes = models.TextField(_('Notes'), blank=True)
    is_active = models.BooleanField(_('Is Active'), default=True) # Track currently taken meds

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.dosage}) for {self.patient_profile.user.email}"

    class Meta:
        verbose_name = _("Medication Record")
        verbose_name_plural = _("Medication Records")
        ordering = ['-start_date', 'name']

    def clean(self):
        # Example validation: End date should not be before start date
        if self.end_date and self.start_date and self.end_date < self.start_date:
            raise ValidationError(_("End date cannot be before start date."))