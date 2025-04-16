from django.contrib import admin
from .models import PatientProfile, Medication

@admin.register(PatientProfile)
class PatientProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'date_of_birth', 'created_at')
    search_fields = ('user__email', 'user__username')
    raw_id_fields = ('user',) # User 선택 편의성

@admin.register(Medication)
class MedicationAdmin(admin.ModelAdmin):
    list_display = ('name', 'patient_profile', 'start_date', 'end_date', 'is_active')
    list_filter = ('is_active', 'start_date')
    search_fields = ('name', 'patient_profile__user__email')
    raw_id_fields = ('patient_profile',)