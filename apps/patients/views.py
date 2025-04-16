# apps/patients/views.py
from rest_framework import viewsets, permissions, status
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from .models import PatientProfile, Medication
from .serializers import PatientProfileSerializer, MedicationSerializer, PatientMedicationCreateSerializer
from apps.users.permissions import IsPatient, IsPatientOwner # 커스텀 권한

class PatientProfileViewSet(viewsets.ReadOnlyModelViewSet): # 보통 조회만 허용, 수정은 /users/me/ 활용
    """ API endpoint for patient profiles. """
    serializer_class = PatientProfileSerializer
    permission_classes = [permissions.IsAuthenticated] # 접근 권한은 get_queryset에서 처리

    def get_queryset(self):
        user = self.request.user
        if user.is_staff: # 관리자는 모든 프로필 조회 가능
            return PatientProfile.objects.select_related('user').prefetch_related('medications').all()
        elif user.role == 'PATIENT': # 환자는 자신의 프로필만 조회 가능
            return PatientProfile.objects.select_related('user').prefetch_related('medications').filter(user=user)
        # 의료인은 담당 환자만 볼 수 있도록 추가 로직 필요 (예: 별도 '담당의-환자' 모델)
        # elif user.role == 'CLINICIAN':
        #     return PatientProfile.objects.filter(담당의=user)...
        return PatientProfile.objects.none()

class PatientMedicationViewSet(viewsets.ModelViewSet):
    """ API endpoint for managing medications for the logged-in patient. """
    permission_classes = [permissions.IsAuthenticated, IsPatient] # 환자만 자신의 약 관리

    def get_serializer_class(self):
        if self.action in ['create', 'update', 'partial_update']:
            return PatientMedicationCreateSerializer
        return MedicationSerializer

    def get_queryset(self):
        # 현재 로그인한 환자의 약 기록만 반환
        # PatientProfile 이 없는 환자는 접근 불가 (IsPatient 퍼미션에서 걸러짐)
        try:
            patient_profile = self.request.user.patient_profile
            return Medication.objects.filter(patient_profile=patient_profile)
        except PatientProfile.DoesNotExist:
             return Medication.objects.none()

    def perform_create(self, serializer):
        # 생성 시 현재 로그인한 환자의 프로필을 자동으로 설정
        try:
             patient_profile = self.request.user.patient_profile
             serializer.save(patient_profile=patient_profile)
        except PatientProfile.DoesNotExist:
             # 이론적으로 IsPatient 퍼미션 때문에 여기까지 오지 않음
             from rest_framework.exceptions import PermissionDenied
             raise PermissionDenied("Patient profile not found for this user.")

    # 삭제/수정 시에도 해당 약이 현재 환자의 것인지 확인 (기본 ModelViewSet이 처리하지만, 추가 검증 가능)
    # def get_object(self):
    #     obj = super().get_object()
    #     # 추가 검증 로직 (예: obj.patient_profile.user == self.request.user)
    #     return obj