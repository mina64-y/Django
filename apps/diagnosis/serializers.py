# apps/diagnosis/serializers.py
from rest_framework import serializers
from django.conf import settings
from .models import DiagnosisRequest, DiagnosisResult
from apps.users.serializers import UserSerializer
from apps.patients.models import PatientProfile # PatientProfile 임포트

class DiagnosisResultSerializer(serializers.ModelSerializer):
    """ CT 진단 결과 Serializer """
    # 필요시 *_plot_path, pdf_report_path 를 전체 URL로 변환
    input_image_slice_plot_url = serializers.SerializerMethodField()
    segmentation_map_plot_url = serializers.SerializerMethodField()
    pdf_report_url = serializers.SerializerMethodField()

    class Meta:
        model = DiagnosisResult
        # DB 모델 필드에 맞게 정의
        fields = (
            'request', 'result_summary', 'confidence_score',
            'classification_prediction', 'classification_probability', # 분류 결과 필드 추가 가정
            'segmentation_metrics', 'input_image_slice_plot', 'segmentation_map_plot',
            'pdf_report_path', 'gemini_interpretation', 'error_message', 'completion_timestamp', 'visualization_3d_html_path',
            # URL 필드 추가
            'input_image_slice_plot_url', 'segmentation_map_plot_url', 'pdf_report_url',
            'visualization_3d_html_url',
        )
        read_only_fields = fields # 결과는 읽기 전용

    def _get_media_url(self, obj, field_name):
        """ Helper to get full media URL """
        relative_path = getattr(obj, field_name, None)
        if relative_path:
            request = self.context.get('request')
            if request:
                return request.build_absolute_uri(settings.MEDIA_URL + relative_path)
            return settings.MEDIA_URL + relative_path
        return None

    def get_input_image_slice_plot_url(self, obj):
        return self._get_media_url(obj, 'input_image_slice_plot')

    def get_segmentation_map_plot_url(self, obj):
        return self._get_media_url(obj, 'segmentation_map_plot')

    def get_pdf_report_url(self, obj):
         return self._get_media_url(obj, 'pdf_report_path')

class DiagnosisRequestSerializer(serializers.ModelSerializer):
    """ CT 진단 요청 조회 Serializer """
    requester = UserSerializer(read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    result = DiagnosisResultSerializer(read_only=True) # 결과 포함

    class Meta:
        model = DiagnosisRequest
        fields = ('id', 'requester', 'patient', 'input_data_reference', 'status', 'status_display',
                  'celery_task_id', 'request_timestamp', 'updated_at', 'result')
        read_only_fields = ('id', 'requester', 'status', 'status_display', 'celery_task_id',
                           'request_timestamp', 'updated_at', 'result')


class DiagnosisRequestCreateSerializer(serializers.ModelSerializer):
    """ CT 진단 요청 생성용 Serializer """
    patient = serializers.PrimaryKeyRelatedField(
        queryset=PatientProfile.objects.all() # TODO: 필요시 접근 가능한 환자만 필터링
    )
    # input_data_reference 는 클라이언트가 파일 업로드 후 전달하는 경로/키
    input_data_reference = serializers.CharField(
        max_length=1024, required=True, write_only=True,
        help_text="Path or key of the uploaded NIfTI file (e.g., relative path in media or S3 key)."
    )

    class Meta:
        model = DiagnosisRequest
        fields = ('patient', 'input_data_reference')

    def validate(self, attrs):
        # TODO: input_data_reference 가 유효한 형식인지 검증 (선택적)
        # TODO: patient 가 requester(clinician)의 담당 환자인지 검증 (중요)
        return attrs