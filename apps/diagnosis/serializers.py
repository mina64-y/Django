# apps/diagnosis/serializers.py
from rest_framework import serializers
from .models import DiagnosisRequest, DiagnosisResult
from apps.users.serializers import UserSerializer
from apps.patients.serializers import PatientProfileSerializer

class DiagnosisResultSerializer(serializers.ModelSerializer):
    # request 필드는 기본적으로 pk 만 보여주므로 필요시 depth 또는 nested serializer 사용
    # request = DiagnosisRequestSerializer(read_only=True) # Example nested
    class Meta:
        model = DiagnosisResult
        fields = '__all__' # 필요한 필드만 명시하는 것이 더 안전
        # fields = ('request', 'result_summary', 'confidence_score', 'pdf_report_path', 'gemini_interpretation', 'additional_data', 'error_message', 'completion_timestamp')
        read_only_fields = ('request', 'completion_timestamp', 'error_message') # 결과는 생성 후 수정 불가

class DiagnosisRequestSerializer(serializers.ModelSerializer):
    requester = UserSerializer(read_only=True)
    # patient 필드는 기본적으로 pk 만 보여주므로 필요시 depth 또는 nested serializer 사용
    # patient = PatientProfileSerializer(read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    result = DiagnosisResultSerializer(read_only=True) # 결과가 있으면 포함

    class Meta:
        model = DiagnosisRequest
        fields = ('id', 'requester', 'patient', 'input_data_reference', 'status', 'status_display', 'celery_task_id', 'request_timestamp', 'updated_at', 'result')
        read_only_fields = ('id', 'requester', 'status', 'status_display', 'celery_task_id', 'request_timestamp', 'updated_at', 'result')

class DiagnosisRequestCreateSerializer(serializers.ModelSerializer):
    """ Serializer specifically for creating a new request. """
    # requester 는 view 에서 자동으로 설정됨
    # status, celery_task_id 는 생성 시 설정하지 않음
    class Meta:
        model = DiagnosisRequest
        fields = ('patient', 'input_data_reference') # 생성 시 필요한 필드만 정의
        # patient 필드에 대해 queryset 설정 또는 view에서 처리 필요