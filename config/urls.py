from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView, SpectacularRedocView

urlpatterns = [
    path('admin/', admin.site.urls),

    # API V1 URLs - '/api/v1/' 접두사 사용
    path('api/v1/', include('config.api_v1_urls')),

    # API 스키마 및 문서 URL
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('api/docs/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'), # Swagger UI
    path('api/redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),       # Redoc UI
]

# 개발 환경에서 미디어 파일 서빙 설정
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    # 개발 환경에서 staticfiles 폴더 서빙이 필요하다면 추가 (보통 DEBUG=True 시 Django가 처리)
    # urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)