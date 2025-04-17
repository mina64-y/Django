# apps/multi_omics/views.py
from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import MultiOmicsRequest, MultiOmicsResult
from .serializers import (
    MultiOmicsRequestSerializer, MultiOmicsResultSerializer, MultiOmicsRequestCreateSerializer
)
from .tasks import run_multi_omics_prediction_task
from apps.users.permissions import IsClinician # 의료인 권한
from django.http import FileResponse, Http404
import os
from django.core.files.storage import default_storage

class MultiOmicsRequestViewSet(viewsets.ModelViewSet):
    """ Multi-omics 예측 요청 생성 및 조회 API """
    queryset = MultiOmicsRequest.objects.all()
    # serializer_class = MultiOmicsRequestSerializer # Use get_serializer_class

    def get_serializer_class(self):
        if self.action == 'create':
            return MultiOmicsRequestCreateSerializer
        return MultiOmicsRequestSerializer

    def get_permissions(self):
        if self.action == 'create':
            # 요청 생성은 의료인만 가능
            self.permission_classes = [permissions.IsAuthenticated, IsClinician]
        elif self.action in ['retrieve', 'list']:
            # 조회는 관련자(요청자, 환자) 또는 관리자만 가능하도록 권한 설정 필요
            self.permission_classes = [permissions.IsAuthenticated] # TODO: Implement IsOwnerOrClinician
        else:
            # 수정/삭제는 관리자만 가능하도록 제한
            self.permission_classes = [permissions.IsAdminUser]
        return super().get_permissions()

    def perform_create(self, serializer):
        """ 요청 생성 시 requester 설정 및 Celery 작업 실행 """
        # TODO: 입력 데이터 참조(gene_data_ref 등)가 S3 경로 등 유효한 값인지 확인 필요
        # 현재는 serializer 에서 받은 값 그대로 저장
        instance = serializer.save(requester=self.request.user)
        # Celery 작업 비동기 실행
        task_result = run_multi_omics_prediction_task.delay(instance.id)
        print(f"Multi-omics prediction task ({task_result.id}) queued for request {instance.id}")
        # Task ID 를 즉시 저장하려면:
        # instance.celery_task_id = task_result.id
        # instance.save(update_fields=['celery_task_id'])

    def get_queryset(self):
        """ 사용자 역할에 따라 접근 가능한 요청 필터링 """
        user = self.request.user
        if user.is_staff or user.role == 'ADMIN':
            return MultiOmicsRequest.objects.all()
        elif user.role == 'CLINICIAN':
            return MultiOmicsRequest.objects.filter(requester=user)
        elif user.role == 'PATIENT':
            return MultiOmicsRequest.objects.filter(patient__user=user)
        return MultiOmicsRequest.objects.none()

    @action(detail=True, methods=['get'])
    def status(self, request, pk=None):
        """ 특정 요청 상태 조회 """
        req_instance = self.get_object()
        return Response({'status': req_instance.get_status_display()})

class MultiOmicsResultViewSet(viewsets.ReadOnlyModelViewSet):
    """ Multi-omics 예측 결과 조회 API """
    queryset = MultiOmicsResult.objects.select_related('request', 'request__patient__user', 'request__requester').all()
    serializer_class = MultiOmicsResultSerializer
    permission_classes = [permissions.IsAuthenticated] # TODO: Implement IsOwnerOrClinician

    def get_queryset(self):
        """ 사용자 역할에 따라 접근 가능한 결과 필터링 """
        user = self.request.user
        # MultiOmicsRequestViewSet의 get_queryset 로직과 유사하게 구현
        if user.is_staff or user.role == 'ADMIN':
            return MultiOmicsResult.objects.select_related('request', 'request__patient__user', 'request__requester').all()
        elif user.role == 'CLINICIAN':
            return MultiOmicsResult.objects.filter(request__requester=user).select_related('request', 'request__patient__user', 'request__requester')
        elif user.role == 'PATIENT':
             return MultiOmicsResult.objects.filter(request__patient__user=user).select_related('request', 'request__patient__user', 'request__requester')
        return MultiOmicsResult.objects.none()

    @action(detail=True, methods=['get'], url_path='download-pdf')
    def download_pdf(self, request, pk=None):
        """ 생성된 Multi-omics PDF 보고서 다운로드 """
        result = self.get_object() # 결과 객체 (pk는 request_id)
        if not result.pdf_report_path:
            raise Http404("PDF report is not available for this result.")

        pdf_relative_path = result.pdf_report_path
        try:
            if not default_storage.exists(pdf_relative_path):
                 raise Http404(f"PDF file not found at path: {pdf_relative_path}")

            pdf_file = default_storage.open(pdf_relative_path, 'rb')
            response = FileResponse(pdf_file, content_type='application/pdf')
            response['Content-Disposition'] = f'inline; filename="multi_omics_report_{result.request_id}.pdf"'
            # 강제 다운로드: response['Content-Disposition'] = f'attachment; filename=...'
            return response
        except FileNotFoundError:
            raise Http404("PDF file not found.")
        except Exception as e:
             print(f"Error serving Multi-omics PDF for result {pk}: {e}")
             return Response({"error": "Could not serve PDF file."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)