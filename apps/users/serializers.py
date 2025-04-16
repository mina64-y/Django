# apps/users/serializers.py
from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from django.core import exceptions as django_exceptions

User = get_user_model()

class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])
    password2 = serializers.CharField(write_only=True, required=True, label="Password Confirmation")

    class Meta:
        model = User
        # username 필드 제거 (email 기반) 또는 필요시 포함
        fields = ('email', 'username', 'password', 'password2', 'first_name', 'last_name', 'role')
        extra_kwargs = {
            'first_name': {'required': False},
            'last_name': {'required': False},
            'role': {'required': True} # 회원가입 시 역할 명시적 요구
        }

    def validate(self, attrs):
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError({"password": "Password fields didn't match."})
        # email 중복 검사는 ModelSerializer가 자동으로 처리 (unique=True 설정 시)
        return attrs

    def create(self, validated_data):
        # role은 validated_data에 포함되어 있음
        user = User.objects.create_user(
            username=validated_data['username'], # username도 저장
            email=validated_data['email'],
            password=validated_data['password'],
            first_name=validated_data.get('first_name', ''),
            last_name=validated_data.get('last_name', ''),
            role=validated_data['role']
        )
        # user.set_password(validated_data['password']) # create_user가 처리해줌
        # user.save() # create_user가 처리해줌
        return user

class UserSerializer(serializers.ModelSerializer):
    """ Serializer for retrieving user information """
    role_display = serializers.CharField(source='get_role_display', read_only=True) # 역할 표시 이름

    class Meta:
        model = User
        fields = ('id', 'email', 'username', 'first_name', 'last_name', 'role', 'role_display', 'is_staff', 'date_joined')
        read_only_fields = ('id', 'email', 'username', 'role', 'role_display', 'is_staff', 'date_joined') # 프로필 수정은 별도 엔드포인트 권장