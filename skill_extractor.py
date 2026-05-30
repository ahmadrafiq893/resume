# ==========================================
# AI Resume Analyzer - Skill Extractor
# ==========================================

class SkillExtractor:
    def __init__(self):
        self.skills_db = [
            "python", "java", "c++", "c#", "javascript", "typescript",
            "sql", "mysql", "postgresql", "mongodb", "nosql",
            "html", "css", "react", "angular", "vue", "nodejs", "express",
            "django", "flask", "spring", "laravel", "php",
            "machine learning", "deep learning", "tensorflow", "pytorch", "keras",
            "data analysis", "data visualization", "pandas", "numpy", "matplotlib",
            "power bi", "tableau", "excel", "statistics",
            "aws", "azure", "google cloud", "docker", "kubernetes", "jenkins", "git",
            "agile", "scrum", "selenium", "manual testing",
            "nlp", "computer vision", "llm", "transformers",
            "communication", "leadership", "problem solving", "teamwork", "time management"
        ]
    
    def extract_skills(self, text):
        text = text.lower()
        found_skills = []
        for skill in self.skills_db:
            if skill in text:
                found_skills.append(skill.title())
        return list(set(found_skills))[:20]