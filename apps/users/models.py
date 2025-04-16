from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils.translation import gettext_lazy as _

class UserRoleChoices(models.TextChoices):
    CLINICIAN = 'CLINICIAN', _('Clinician') # 의료인
    PATIENT = 'PATIENT', _('Patient')     # 환자
    ADMIN = 'ADMIN', _('Admin')         # 관리자

class User(AbstractUser):
    # Add additional fields if needed, e.g., email verification, profile picture
    email = models.EmailField(_('email address'), unique=True) # Make email unique and required for login
    role = models.CharField(
        _('role'),
        max_length=20,
        choices=UserRoleChoices.choices,
        default=UserRoleChoices.PATIENT # Default role, adjust as needed
    )

    # Use email for login instead of username
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username'] # Keep username for compatibility or remove if not needed

    def __str__(self):
        return self.email