from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import PatientProfileViewSet, PatientMedicationViewSet

app_name = 'patients'

router = DefaultRouter()
# '/api/v1/profiles/' (환자 프로필 조회 - 주로 현재 사용자 것만)
router.register(r'profiles', PatientProfileViewSet, basename='patientprofile')
# '/api/v1/medications/' (현재 환자의 약 관리)
router.register(r'medications', PatientMedicationViewSet, basename='medication')

urlpatterns = [
    path('', include(router.urls)),
]