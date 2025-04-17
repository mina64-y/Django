# apps/multi_omics/serializers.py
from rest_framework import serializers
from django.conf import settings
from .models import MultiOmicsRequest, MultiOmicsResult
from apps.users.serializers import UserSerializer # 요청자 정보 표시용
from apps.patients.models import PatientProfile # Patient 외래키 필드용

class MultiOmicsResultSerializer(serializers.ModelSerializer):
    """ Multi-omics 예측 결과 Serializer """

    # 각 플롯 및 PDF 파일의 전체 URL을 생성하기 위한 SerializerMethodField 정의
    gene_plot_url = serializers.SerializerMethodField()
    protein_plot_url = serializers.SerializerMethodField()
    methylation_plot_url = serializers.SerializerMethodField()
    cnv_plot_url = serializers.SerializerMethodField()
    gauge_plot_url = serializers.SerializerMethodField()
    radar_plot_url = serializers.SerializerMethodField()
    pdf_report_url = serializers.SerializerMethodField() # PDF URL 필드 추가

    class Meta:
        model = MultiOmicsResult
        # 모델에 정의된 모든 필드를 포함하고, URL 필드들을 추가합니다.
        # request 필드는 보통 결과 조회 시 필요 없으므로 exclude 하거나 fields 에서 제외합니다.
        fields = (
            'ensemble_probability',
            'ensemble_prediction',
            'individual_probabilities',
            'threshold_used',
            'gene_plot_path', # 상대 경로 (디버깅/참고용)
            'protein_plot_path',
            'methylation_plot_path',
            'cnv_plot_path',
            'gauge_plot_path',
            'radar_plot_path',
            'pdf_report_path', # PDF 상대 경로 추가
            'gemini_interpretation',
            'error_message',
            'completion_timestamp',
            # 생성된 URL 필드들
            'gene_plot_url',
            'protein_plot_url',
            'methylation_plot_url',
            'cnv_plot_url',
            'gauge_plot_url',
            'radar_plot_url',
            'pdf_report_url', # PDF URL 추가
        )
        # 결과는 보통 생성 후 수정하지 않으므로 모든 필드를 읽기 전용으로 설정 가능
        read_only_fields = fields

    def _get_media_url(self, obj, field_name):
        """
        주어진 필드 이름에 해당하는 상대 경로 값으로 전체 미디어 URL을 생성하는 헬퍼 함수.
        DRF ViewSet context 에 request 가 있으면 절대 URL 생성, 없으면 MEDIA_URL 기반 URL 생성.
        """
        relative_path = getattr(obj, field_name, None)
        if relative_path:
            request = self.context.get('request')
            # ViewSet 을 통해 Serializer 가 호출될 때 request 객체가 context 에 주입됨
            if request:
                try:
                    # build_absolute_uri 를 사용하여 현재 요청의 스킴/호스트 포함 URL 생성
                    return request.build_absolute_uri(settings.MEDIA_URL + relative_path)
                except Exception:
                    # request 객체가 build_absolute_uri 를 지원하지 않는 예외적 경우 대비
                    pass
            # request 객체가 없거나 절대 URL 생성 실패 시 MEDIA_URL 기준 URL 생성
            # settings.MEDIA_URL 이 '/' 로 시작하는 상대 경로일 경우 주의 필요
            # 운영 환경에서는 MEDIA_URL 앞에 도메인을 붙여 완전한 URL 이 되도록 설정하는 것이 좋음
            # 예: MEDIA_URL = 'https://your-domain.com/media/'
            media_url_base = settings.MEDIA_URL.rstrip('/')
            file_path_clean = relative_path.lstrip('/')
            return f"{media_url_base}/{file_path_clean}"
        return None # 해당 필드 값이 없으면 None 반환

    # 각 필드에 대한 URL 생성 메소드 정의
    def get_gene_plot_url(self, obj):
        return self._get_media_url(obj, 'gene_plot_path')

    def get_protein_plot_url(self, obj):
        return self._get_media_url(obj, 'protein_plot_path')

    def get_methylation_plot_url(self, obj):
        return self._get_media_url(obj, 'methylation_plot_path')

    def get_cnv_plot_url(self, obj):
        return self._get_media_url(obj, 'cnv_plot_path')

    def get_gauge_plot_url(self, obj):
        return self._get_media_url(obj, 'gauge_plot_path')

    def get_radar_plot_url(self, obj):
        return self._get_media_url(obj, 'radar_plot_path')

    def get_pdf_report_url(self, obj):
        # 새로 추가된 PDF 경로에 대한 URL 생성
        return self._get_media_url(obj, 'pdf_report_path')


class MultiOmicsRequestSerializer(serializers.ModelSerializer):
    """ Multi-omics 예측 요청 조회 Serializer """
    requester = UserSerializer(read_only=True) # 요청자 상세 정보 포함
    # patient = PatientProfileSerializer(read_only=True) # 필요시 환자 상세 정보 포함
    status_display = serializers.CharField(source='get_status_display', read_only=True) # 상태 표시 이름
    result = MultiOmicsResultSerializer(read_only=True) # 생성된 결과 포함

    class Meta:
        model = MultiOmicsRequest
        # 모델의 모든 필드를 포함하거나 필요한 필드만 명시
        fields = '__all__'
        # 읽기 전용 필드 설정
        read_only_fields = ('id', 'requester', 'status', 'status_display',
                           'celery_task_id', 'request_timestamp', 'updated_at', 'result')

class MultiOmicsRequestCreateSerializer(serializers.ModelSerializer):
    """ Multi-omics 예측 요청 생성용 Serializer """
    # patient 필드는 PatientProfile의 PK를 받도록 설정
    patient = serializers.PrimaryKeyRelatedField(
        queryset=PatientProfile.objects.all(), # TODO: 필요시 접근 가능한 환자만 필터링 (View 에서 처리 가능)
        help_text="예측을 요청할 환자의 프로필 ID"
    )

    # 입력 데이터 참조 필드들 (클라이언트가 파일 업로드 후 경로 전달 방식 가정)
    gene_data_ref = serializers.CharField(max_length=1024, required=False, allow_blank=True, allow_null=True, help_text="유전자 데이터 파일 경로/키")
    protein_data_ref = serializers.CharField(max_length=1024, required=False, allow_blank=True, allow_null=True, help_text="단백질 데이터 파일 경로/키")
    methylation_data_ref = serializers.CharField(max_length=1024, required=False, allow_blank=True, allow_null=True, help_text="메틸화 데이터 파일 경로/키")
    cnv_data_ref = serializers.CharField(max_length=1024, required=False, allow_blank=True, allow_null=True, help_text="CNV 데이터 파일 경로/키")

    class Meta:
        model = MultiOmicsRequest
        # 요청 생성 시 필요한 필드만 포함
        fields = ('patient', 'gene_data_ref', 'protein_data_ref', 'methylation_data_ref', 'cnv_data_ref')

    def validate(self, attrs):
        # 최소 하나 이상의 데이터 참조가 있는지 확인하는 유효성 검사
        refs = [attrs.get('gene_data_ref'), attrs.get('protein_data_ref'),
                attrs.get('methylation_data_ref'), attrs.get('cnv_data_ref')]
        if not any(ref for ref in refs if ref): # None 이나 빈 문자열이 아닌 값이 하나라도 있는지 확인
            raise serializers.ValidationError("최소 하나 이상의 데이터 파일 참조(경로/키)를 제공해야 합니다.")

        # TODO: View 레벨에서, 요청하는 의료인(request.user)이 attrs['patient'] 에 접근 권한이 있는지 확인하는 로직 추가 필요
        return attrs