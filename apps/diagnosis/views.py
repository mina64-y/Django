# apps/diagnosis/views.py
from rest_framework import viewsets, permissions, status, generics
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.http import FileResponse, Http404
from django.conf import settings
import os

from .models import DiagnosisRequest, DiagnosisResult
from .serializers import ( # Serializer 이름 확인 및 임포트
    DiagnosisRequestSerializer, DiagnosisResultSerializer,
    DiagnosisRequestCreateSerializer # CT용 Create Serializer 정의 필요
)
from .tasks import run_ct_diagnosis_task # CT용 Task 임포트
from apps.users.permissions import IsClinician # 의료인 권한

class DiagnosisRequestViewSet(viewsets.ModelViewSet):
    """ CT 진단 요청 생성 및 조회 API """
    queryset = DiagnosisRequest.objects.all()

    def get_serializer_class(self):
        if self.action == 'create':
            return DiagnosisRequestCreateSerializer # CT용 Create Serializer 사용
        return DiagnosisRequestSerializer

    def get_permissions(self):
        """ 액션에 따른 권한 부여 """
        if self.action == 'create':
            self.permission_classes = [permissions.IsAuthenticated, IsClinician]
        elif self.action in ['retrieve', 'list', 'status']: # 상태 조회 추가
            # TODO: 환자 본인 또는 관련 의료인만 접근 가능하도록 IsOwnerOrClinician 권한 구현 필요
            self.permission_classes = [permissions.IsAuthenticated]
        else:
            self.permission_classes = [permissions.IsAdminUser] # 수정/삭제는 관리자만
        return super().get_permissions()

    def perform_create(self, serializer):
        """ 요청 생성 시 requester 설정 및 Celery 작업 실행 """
        # Serializer 는 patient 와 input_data_reference (NIfTI 파일 상대 경로) 를 받음
        # 파일 업로드 및 경로 생성/전달 로직은 이 View 이전 단계 또는 클라이언트 측에서 처리되어야 함
        # 예: 클라이언트가 S3에 업로드 후 S3 key 를 input_data_reference 로 전달
        # 예: 별도 파일 업로드 API 에서 파일을 media 에 저장 후 그 경로를 input_data_reference 로 전달
        instance = serializer.save(requester=self.request.user, status=DiagnosisRequestStatus.PENDING) # 상태 PENDING 으로 생성
        # CT 진단 Celery 작업 비동기 실행
        task_result = run_ct_diagnosis_task.delay(instance.id)
        print(f"CT Diagnosis task ({task_result.id}) queued for request {instance.id}")
        # 생성 직후 task id 업데이트 (선택적)
        # instance.celery_task_id = task_result.id
        # instance.save(update_fields=['celery_task_id'])

    def get_queryset(self):
        """ 사용자 역할 기반 필터링 """
        user = self.request.user
        if user.is_staff or user.role == 'ADMIN':
            return DiagnosisRequest.objects.all()
        elif user.role == 'CLINICIAN':
            return DiagnosisRequest.objects.filter(requester=user)
        elif user.role == 'PATIENT':
            return DiagnosisRequest.objects.filter(patient__user=user)
        return DiagnosisRequest.objects.none()

    @action(detail=True, methods=['get'])
    def status(self, request, pk=None):
        """ 특정 진단 요청 상태 조회 """
        diag_request = self.get_object()
        # 필요시 Celery 백엔드에서 실제 작업 상태 조회 로직 추가 가능
        # from celery.result import AsyncResult
        # task_status = AsyncResult(diag_request.celery_task_id).state if diag_request.celery_task_id else 'N/A'
        return Response({
            'db_status': diag_request.get_status_display(),
            'status_code': diag_request.status # 상태 코드 직접 반환
            # 'celery_task_status': task_status
        })

class DiagnosisResultViewSet(viewsets.ReadOnlyModelViewSet):
    """ CT 진단 결과 조회 API """
    queryset = DiagnosisResult.objects.select_related('request', 'request__patient__user', 'request__requester').all()
    serializer_class = DiagnosisResultSerializer
    # TODO: 접근 권한 설정 (IsOwnerOrClinician 등)
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        """ 사용자 역할 기반 필터링 """
        # 이전 답변의 MultiOmicsResultViewSet.get_queryset 로직과 동일하게 구현
        user = self.request.user
        if user.is_staff or user.role == 'ADMIN':
            return DiagnosisResult.objects.select_related('request', 'request__patient__user', 'request__requester').all()
        elif user.role == 'CLINICIAN':
            return DiagnosisResult.objects.filter(request__requester=user).select_related('request', 'request__patient__user', 'request__requester')
        elif user.role == 'PATIENT':
             return DiagnosisResult.objects.filter(request__patient__user=user).select_related('request', 'request__patient__user', 'request__requester')
        return DiagnosisResult.objects.none()

    @action(detail=True, methods=['get'], url_path='download-pdf')
    def download_pdf(self, request, pk=None):
        """ 생성된 PDF 보고서 다운로드 """
        result = self.get_object() # 결과 객체 가져오기 (pk는 request_id와 동일)
        if not result.pdf_report_path:
            raise Http404("PDF report is not available for this result.")

        # DB에 저장된 상대 경로 사용
        pdf_relative_path = result.pdf_report_path
        try:
            # default_storage 사용하여 파일 열기 (S3 등 다른 스토리지 사용 시에도 동일하게 작동)
            if not default_storage.exists(pdf_relative_path):
                 raise Http404(f"PDF file not found at path: {pdf_relative_path}")

            pdf_file = default_storage.open(pdf_relative_path, 'rb')
            response = FileResponse(pdf_file, content_type='application/pdf')
            # 파일명 제안
            response['Content-Disposition'] = f'inline; filename="ct_diagnosis_report_{result.request_id}.pdf"'
            # 강제 다운로드: response['Content-Disposition'] = f'attachment; filename=...'
            return response
        except FileNotFoundError:
            raise Http404("PDF file not found.")
        except Exception as e:
             print(f"Error serving PDF for result {pk}: {e}")
             return Response({"error": "Could not serve PDF file."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)