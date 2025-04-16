import time
import os
from celery import shared_task
from django.utils import timezone
from django.conf import settings
import requests # For Gemini API
from .models import DiagnosisRequest, DiagnosisResult, DiagnosisRequestStatus
from .pdf_utils import generate_diagnosis_pdf # Assumes pdf_utils.py exists
# Import your actual AI model loading and prediction functions
# from .ai_models import load_pancreas_model, predict_pancreas_cancer

@shared_task(bind=True)
def run_diagnosis_task(self, request_id):
    """
    Celery task to run AI diagnosis, generate PDF, and get Gemini interpretation.
    """
    try:
        request = DiagnosisRequest.objects.get(id=request_id)
        request.celery_task_id = self.request.id
        request.status = DiagnosisRequestStatus.PROCESSING
        request.save(update_fields=['status', 'celery_task_id'])
        print(f"[Task {self.request.id}] Processing Diagnosis Request {request_id}...")

        # --- 1. Load Input Data (replace with actual logic) ---
        # This depends on how input_data_reference is used (e.g., download from S3)
        print(f"   Input data reference: {request.input_data_reference}")
        # Placeholder: Assume data is loaded into memory or accessible path
        input_data = None # Load your data here (e.g., nibabel image)
        # Simulate loading time
        time.sleep(2)
        if not input_data: # Simulate failure if needed
             # raise ValueError("Failed to load input data.")
             pass # Continue with placeholder for now

        # --- 2. Run AI Model Prediction (Replace with actual model) ---
        print("   Running AI model prediction...")
        # model = load_pancreas_model()
        # prediction, confidence, structured_output = predict_pancreas_cancer(model, input_data)
        # Simulate prediction
        time.sleep(10) # Simulate long AI process
        prediction = "Positive" # Placeholder
        confidence = 0.85 # Placeholder
        structured_output = {"tumor_volume_cc": 12.5, "location": "Head"} # Placeholder

        print(f"   AI Prediction: {prediction}, Confidence: {confidence:.2f}")

        # --- 3. Generate PDF Report ---
        print("   Generating PDF report...")
        pdf_context = {
            'request': request,
            'patient': request.patient,
            'requester': request.requester,
            'prediction': prediction,
            'confidence': confidence,
            'structured_output': structured_output,
            'generation_time': timezone.now(),
        }
        # pdf_relative_path = generate_diagnosis_pdf(request_id, pdf_context) # Returns path relative to MEDIA_ROOT
        pdf_relative_path = f"diagnosis_reports/report_{request_id}.pdf" # Placeholder
        print(f"   PDF generated at: {pdf_relative_path}")

        # --- 4. Get Gemini Interpretation ---
        print("   Requesting Gemini interpretation...")
        gemini_interpretation = "Gemini interpretation is currently unavailable." # Default/fallback
        gemini_api_key = settings.GEMINI_API_KEY
        if gemini_api_key and prediction: # Only request if API key exists and prediction was made
            try:
                # Construct a detailed prompt for Gemini
                prompt = f"""
                Analyze the following pancreatic cancer diagnosis result for patient {request.patient.id}.
                Clinician: {request.requester.get_full_name()}
                Input Data Reference: {request.input_data_reference}
                AI Model Prediction: {prediction}
                Confidence Score: {confidence:.2f}
                Additional Model Output: {structured_output}

                Provide a concise interpretation for the clinician, highlighting key findings and potential implications.
                Keep the language professional and appropriate for a medical context.
                """
                # Replace with actual Gemini API call logic using 'requests' or google-generativeai library
                # response = requests.post(...)
                # gemini_interpretation = response.json().get(...)
                time.sleep(3) # Simulate API call
                gemini_interpretation = f"Based on the analysis, the result is '{prediction}' with a confidence of {confidence*100:.1f}%. {structured_output.get('location','N/A')} location noted. Further clinical correlation recommended." # Placeholder
                print("   Gemini interpretation received.")
            except Exception as gemini_err:
                print(f"   ERROR calling Gemini API: {gemini_err}")
                # Keep the default interpretation
        else:
            print("   Skipping Gemini API call (No API key or prediction).")


        # --- 5. Save Result ---
        print("   Saving final result...")
        result, created = DiagnosisResult.objects.update_or_create(
            request=request,
            defaults={
                'result_summary': prediction,
                'confidence_score': confidence,
                'pdf_report_path': pdf_relative_path,
                'gemini_interpretation': gemini_interpretation,
                'additional_data': structured_output,
                'error_message': "", # Clear previous errors if any
                'completion_timestamp': timezone.now()
            }
        )
        request.status = DiagnosisRequestStatus.COMPLETED
        request.save(update_fields=['status'])
        print(f"[Task {self.request.id}] Diagnosis Request {request_id} COMPLETED.")
        return result.request_id # Return something meaningful

    except DiagnosisRequest.DoesNotExist:
        print(f"[Task {self.request.id}] ERROR: DiagnosisRequest {request_id} not found.")
        # No request object to update status on
        return None
    except Exception as e:
        print(f"[Task {self.request.id}] FAILED processing Diagnosis Request {request_id}: {e}")
        print(traceback.format_exc())
        # Try to update the request status to FAILED and save the error
        try:
            request = DiagnosisRequest.objects.get(id=request_id)
            request.status = DiagnosisRequestStatus.FAILED
            request.save(update_fields=['status'])
            DiagnosisResult.objects.update_or_create(
                request=request,
                defaults={
                    'error_message': f"{type(e).__name__}: {str(e)}",
                    'completion_timestamp': timezone.now()
                    # Clear other fields or leave them as they were
                }
            )
        except Exception as update_err:
            print(f"   ERROR: Could not update request/result status to FAILED: {update_err}")
        # Re-raise the exception so Celery knows the task failed
        raise e