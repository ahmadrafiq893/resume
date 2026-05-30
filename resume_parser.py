# ==========================================
# AI Resume Analyzer - Resume Parser
# ==========================================

import re
import fitz  # PyMuPDF

class ResumeParser:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def extract_text(self):
        text = ""
        try:
            pdf = fitz.open(self.file_path)
            for page in pdf:
                text += page.get_text()
            pdf.close()
            return text
        except Exception as e:
            print("PDF Error:", e)
            return ""
    
    def clean_text(self, text):
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s,.-]", "", text)
        return text.strip()
    
    def get_resume_text(self):
        text = self.extract_text()
        text = self.clean_text(text)
        return text