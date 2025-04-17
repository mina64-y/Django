# apps/multi_omics/apps.py
from django.apps import AppConfig

class MultiOmicsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.multi_omics' # 앱 경로 포함
    verbose_name = 'Multi-omics Prediction' # 관리자 페이지 표시 이름