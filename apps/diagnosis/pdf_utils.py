# apps/diagnosis/pdf_utils.py
import os
from django.conf import settings
from django.template.loader import render_to_string
# from weasyprint import HTML # Weasyprint 설치 후 사용
import traceback

def generate_diagnosis_pdf(request_id, context):
    """ Generate PDF report and return path relative to MEDIA_ROOT """
    try:
        # 1. 템플릿 렌더링 (템플릿 파일 경로 확인 필요)
        # template_path = 'diagnosis/pdf_report_template.html' # 예시 경로
        # html_string = render_to_string(template_path, context)

        # 2. PDF 생성
        # html = HTML(string=html_string, base_url=settings.MEDIA_ROOT) # 이미지 등 상대 경로 기준
        # pdf_filename = f"diagnosis_report_{request_id}.pdf"
        # pdf_folder = os.path.join(settings.MEDIA_ROOT, 'diagnosis_reports')
        # os.makedirs(pdf_folder, exist_ok=True)
        # pdf_full_path = os.path.join(pdf_folder, pdf_filename)
        # html.write_pdf(pdf_full_path)

        # Placeholder implementation
        pdf_relative_path = f"diagnosis_reports/report_{request_id}.pdf"
        pdf_folder = os.path.join(settings.MEDIA_ROOT, 'diagnosis_reports')
        os.makedirs(pdf_folder, exist_ok=True)
        pdf_full_path = os.path.join(pdf_folder, f"report_{request_id}.pdf")
        with open(pdf_full_path, 'w') as f:
            f.write(f"Placeholder PDF for request {request_id}\n")
            f.write(f"Prediction: {context.get('prediction', 'N/A')}\n")
            f.write(f"Confidence: {context.get('confidence', 'N/A')}\n")

        print(f"Generated placeholder PDF: {pdf_full_path}")
        return pdf_relative_path # MEDIA_ROOT 기준 상대 경로 반환

    except Exception as e:
        print(f"!!!!!!!! ERROR generating PDF for request {request_id} !!!!!!!!")
        print(traceback.format_exc())
        return None