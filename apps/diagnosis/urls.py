# apps/diagnosis/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DiagnosisRequestViewSet, DiagnosisResultViewSet

app_name = 'diagnosis'

router = DefaultRouter()
# '/api/v1/diagnosis/requests/'
router.register(r'diagnosis/requests', DiagnosisRequestViewSet, basename='diagnosisrequest')
# '/api/v1/diagnosis/results/'
router.register(r'diagnosis/results', DiagnosisResultViewSet, basename='diagnosisresult')

urlpatterns = [
    path('', include(router.urls)),
]