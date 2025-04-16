# config/api_v1_urls.py
from django.urls import path, include
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView,
)

app_name = 'api_v1'

urlpatterns = [
    # Authentication (JWT) - '/api/v1/auth/token/' 형태로 접근
    path('auth/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('auth/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('auth/token/verify/', TokenVerifyView.as_view(), name='token_verify'),

    # Include app-specific API URLs
    path('', include('apps.users.urls')),     # 예: /api/v1/users/register/
    path('', include('apps.patients.urls')),   # 예: /api/v1/patients/
    path('', include('apps.diagnosis.urls')),  # 예: /api/v1/diagnosis/requests/
]