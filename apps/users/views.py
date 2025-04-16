# apps/users/views.py
from rest_framework import generics, permissions, status
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from .serializers import UserRegistrationSerializer, UserSerializer

User = get_user_model()

class UserRegistrationView(generics.CreateAPIView):
    serializer_class = UserRegistrationSerializer
    permission_classes = [permissions.AllowAny] # 누구나 회원가입 가능

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        # 생성된 사용자 정보 일부 반환 (비밀번호 제외)
        user_data = UserSerializer(user, context=self.get_serializer_context()).data
        headers = self.get_success_headers(user_data)
        return Response(user_data, status=status.HTTP_201_CREATED, headers=headers)


class UserProfileView(generics.RetrieveUpdateAPIView):
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_object(self):
        # 현재 로그인한 사용자 정보 반환
        return self.request.user

    # 업데이트는 필요에 따라 제한적으로 구현 (예: 이름 변경)
    # def perform_update(self, serializer):
    #     serializer.save()