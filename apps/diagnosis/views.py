from rest_framework import viewsets, permissions, status, generics
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from django.http import FileResponse, Http404
from django.conf import settings
import os

from .models import DiagnosisRequest, DiagnosisResult
from .serializers import DiagnosisRequestSerializer, DiagnosisResultSerializer, DiagnosisRequestCreateSerializer
from .tasks import run_diagnosis_task
from apps.users.permissions import IsClinician # Example custom permission

class DiagnosisRequestViewSet(viewsets.ModelViewSet):
    """
    API endpoint for creating and viewing diagnosis requests.
    Clinicians can create requests. Involved clinicians/patients can view.
    """
    queryset = DiagnosisRequest.objects.all()
    # serializer_class = DiagnosisRequestSerializer # Use get_serializer_class

    def get_serializer_class(self):
        if self.action == 'create':
            return DiagnosisRequestCreateSerializer
        return DiagnosisRequestSerializer

    def get_permissions(self):
        """ Assign permissions based on action. """
        if self.action == 'create':
            self.permission_classes = [permissions.IsAuthenticated, IsClinician]
        # Add more granular permissions for list, retrieve, update, destroy
        elif self.action in ['retrieve', 'list']:
             # Allow clinician or associated patient (needs custom perm)
            self.permission_classes = [permissions.IsAuthenticated] # TODO: Implement IsOwnerOrClinician
        else:
            # Generally, requests shouldn't be modified/deleted easily via API once submitted
            self.permission_classes = [permissions.IsAdminUser]
        return super().get_permissions()

    def perform_create(self, serializer):
        """ On creation, set requester and trigger Celery task. """
        # input_data_reference needs to be handled, e.g., from file upload pre-signed URL logic
        # For now, assume it's passed in the serializer or generated here
        instance = serializer.save(requester=self.request.user)
        # Trigger the Celery task
        task_result = run_diagnosis_task.delay(instance.id)
        # Optionally store task ID immediately (task does it too, but good for immediate feedback)
        # instance.celery_task_id = task_result.id
        # instance.save()
        print(f"Diagnosis task ({task_result.id}) queued for request {instance.id}")

    def get_queryset(self):
        """ Filter requests based on user role. """
        user = self.request.user
        if user.is_staff or user.role == 'ADMIN':
            return DiagnosisRequest.objects.all()
        elif user.role == 'CLINICIAN':
            # Clinicians see requests they made
            return DiagnosisRequest.objects.filter(requester=user)
        elif user.role == 'PATIENT':
            # Patients see requests made for them
            return DiagnosisRequest.objects.filter(patient__user=user)
        return DiagnosisRequest.objects.none() # Should not happen for authenticated users

    @action(detail=True, methods=['get'])
    def status(self, request, pk=None):
        """ Get the status of a specific diagnosis request. """
        diag_request = self.get_object() # Checks permissions
        # TODO: Consider adding Celery task status check if needed
        return Response({'status': diag_request.get_status_display()})


class DiagnosisResultViewSet(viewsets.ReadOnlyModelViewSet):
    """
    API endpoint for viewing diagnosis results.
    Accessible by involved clinician/patient or admin.
    """
    queryset = DiagnosisResult.objects.select_related('request', 'request__patient', 'request__requester').all()
    serializer_class = DiagnosisResultSerializer
    permission_classes = [permissions.IsAuthenticated] # TODO: Implement IsOwnerOrClinician

    # Filter based on user like in DiagnosisRequestViewSet
    def get_queryset(self):
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
        """ Download the generated PDF report for a result. """
        result = self.get_object() # Checks object permissions
        if not result.pdf_report_path:
            raise Http404("PDF report not available for this result.")

        pdf_full_path = os.path.join(settings.MEDIA_ROOT, result.pdf_report_path)

        if not os.path.exists(pdf_full_path):
             raise Http404(f"PDF file not found at path: {result.pdf_report_path}")

        try:
            # Use FileResponse for efficient file serving
            response = FileResponse(open(pdf_full_path, 'rb'), content_type='application/pdf')
            # Suggest a filename for the download
            response['Content-Disposition'] = f'inline; filename="diagnosis_report_{result.request_id}.pdf"'
            # Use 'attachment' instead of 'inline' to force download
            # response['Content-Disposition'] = f'attachment; filename="diagnosis_report_{result.request_id}.pdf"'
            return response
        except FileNotFoundError:
            raise Http404("PDF file not found.")
        except Exception as e:
             print(f"Error serving PDF for result {pk}: {e}")
             return Response({"error": "Could not serve PDF file."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Add similar ViewSets for PatientProfile and Medication if needed