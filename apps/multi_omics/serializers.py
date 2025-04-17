# apps/multi_omics/serializers.py
from rest_framework import serializers
from .models import MultiOmicsRequest, MultiOmicsResult
from apps.users.serializers import UserSerializer
# from apps.patients.serializers import PatientProfileSerializer # 필요시 환자 정보 포함

class MultiOmicsResultSerializer(serializers.ModelSerializer):
    # 필요시 MEDIA_URL 포함하여 전체 URL 반환하도록 SerializerMethodField 사용 가능
    # gene_plot_url = serializers.SerializerMethodField()

    class Meta:
        model = MultiOmicsResult
        exclude = ('request',) # 요청 ID는 URL 이나 부모 serializer 에서 알 수 있으므로 제외 가능

    # def get_gene_plot_url(self, obj):
    #     if obj.gene_plot_path:
    #         return self.context['request'].build_absolute_uri(settings.MEDIA_URL + obj.gene_plot_path)
    #     return None

class MultiOmicsRequestSerializer(serializers.ModelSerializer):
    requester = UserSerializer(read_only=True)
    # patient = PatientProfileSerializer(read_only=True) # 필요시 상세 정보 포함
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    result = MultiOmicsResultSerializer(read_only=True) # 결과 포함

    class Meta:
        model = MultiOmicsRequest
        fields = '__all__' # 필요한 필드 명시 권장
        read_only_fields = ('id', 'requester', 'status', 'status_display',
                           'celery_task_id', 'request_timestamp', 'updated_at', 'result')

class MultiOmicsRequestCreateSerializer(serializers.ModelSerializer):
    """ Multi-omics 예측 요청 생성용 Serializer """
    # patient 필드는 PatientProfile의 PK를 받도록 설정
    patient = serializers.PrimaryKeyRelatedField(
        queryset=PatientProfile.objects.all() # 필요시 queryset 제한 (예: 현재 의료인 담당 환자)
    )

    # 입력 데이터 참조 필드들 (클라이언트가 파일 업로드 후 경로 전달 방식 가정)
    # 이 필드들을 필수로 할지, 어떻게 받을지 API 설계에 따라 결정 필요
    gene_data_ref = serializers.CharField(max_length=1024, required=False, allow_blank=True)
    protein_data_ref = serializers.CharField(max_length=1024, required=False, allow_blank=True)
    methylation_data_ref = serializers.CharField(max_length=1024, required=False, allow_blank=True)
    cnv_data_ref = serializers.CharField(max_length=1024, required=False, allow_blank=True)

    class Meta:
        model = MultiOmicsRequest
        fields = ('patient', 'gene_data_ref', 'protein_data_ref', 'methylation_data_ref', 'cnv_data_ref')

    def validate(self, attrs):
        # 최소 하나 이상의 데이터 참조가 있는지 확인 (선택적)
        refs = [attrs.get('gene_data_ref'), attrs.get('protein_data_ref'),
                attrs.get('methylation_data_ref'), attrs.get('cnv_data_ref')]
        if not any(refs):
            raise serializers.ValidationError("At least one data reference must be provided.")
        # TODO: 전달받은 patient가 requester(clinician)의 담당 환자인지 검증하는 로직 추가 필요
        return attrs