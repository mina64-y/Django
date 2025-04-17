# apps/multi_omics/urls.py
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import MultiOmicsRequestViewSet, MultiOmicsResultViewSet

app_name = 'multi_omics'

router = DefaultRouter()
# '/api/v1/multi-omics/requests/'
router.register(r'multi-omics/requests', MultiOmicsRequestViewSet, basename='multiomicsrequest')
# '/api/v1/multi-omics/results/'
router.register(r'multi-omics/results', MultiOmicsResultViewSet, basename='multiomicsresult')

urlpatterns = [
    path('', include(router.urls)),
]