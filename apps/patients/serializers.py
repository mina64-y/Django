from rest_framework import serializers
from .models import PatientProfile, Medication
from apps.users.serializers import UserSerializer # 사용자 정보 표시용

class MedicationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Medication
        fields = ('id', 'name', 'dosage', 'frequency', 'start_date', 'end_date', 'notes', 'is_active', 'created_at', 'updated_at')
        read_only_fields = ('id', 'created_at', 'updated_at')

class PatientProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True) # 사용자 정보 포함 (읽기 전용)
    medications = MedicationSerializer(many=True, read_only=True) # 복용 약물 목록 포함 (읽기 전용)
    # For creating/updating profile, maybe a simpler serializer without nested user/medications
    # or handle updates through specific fields allowed

    class Meta:
        model = PatientProfile
        fields = ('user', 'date_of_birth', 'medications', 'created_at', 'updated_at') # 필요한 필드 추가
        read_only_fields = ('user', 'medications', 'created_at', 'updated_at')

class PatientMedicationCreateSerializer(serializers.ModelSerializer):
    """ Serializer for creating/updating medication records for the current patient """
    # patient_profile is automatically set in the view

    class Meta:
        model = Medication
        # Exclude patient_profile as it's set automatically
        exclude = ('patient_profile',) # 또는 fields 명시