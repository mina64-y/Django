# apps/users/urls.py
from django.urls import path
from .views import UserRegistrationView, UserProfileView

app_name = 'users'

urlpatterns = [
    path('users/register/', UserRegistrationView.as_view(), name='register'),
    path('users/me/', UserProfileView.as_view(), name='profile'),
    # 비밀번호 변경, 이메일 변경 등 추가 엔드포인트 필요시 구현
]