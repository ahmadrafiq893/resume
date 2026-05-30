# ============================================
# AI Resume Analyzer - Flask API
# ============================================

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import joblib
import numpy as np
from scipy.sparse import hstack
from resume_parser import ResumeParser
from skill_extractor import SkillExtractor

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ============================================
# LOAD MODELS
# ============================================

print("\n" + "="*50)
print("🚀 LOADING TRAINED MODELS...")
print("="*50)

try:
    model = joblib.load("models/best_resume_model.pkl")
    tfidf = joblib.load("models/tfidf.pkl")
    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")
    role_encoder = joblib.load("models/jobrole_encoder.pkl")
    print("✅ All models loaded successfully!")
    MODELS_READY = True
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("⚠️ Please run 'python train_model.py' first!")
    MODELS_READY = False

# Education mapping
EDU_MAP = {
    "High School": 0,
    "Diploma": 1,
    "Bachelor": 2,
    "Master": 3,
    "PhD": 4
}

# Role-specific skills for gap analysis
ROLE_SKILLS = {
    "Data Analyst": ["SQL", "Excel", "Power BI", "Tableau", "Python", "Statistics", "Data Visualization"],
    "Data Scientist": ["Python", "Machine Learning", "SQL", "Statistics", "Deep Learning", "Pandas", "Scikit-learn"],
    "ML Engineer": ["Python", "TensorFlow", "PyTorch", "Docker", "AWS", "Machine Learning", "Git"],
    "AI Engineer": ["Python", "Deep Learning", "NLP", "Computer Vision", "PyTorch", "TensorFlow"],
    "Software Engineer": ["Python", "Java", "Git", "SQL", "Data Structures", "Algorithms"],
    "Frontend Developer": ["JavaScript", "React", "HTML", "CSS", "Angular", "Bootstrap"],
    "Backend Developer": ["Python", "Java", "SQL", "Django", "Flask", "Node.js"],
    "Full Stack Developer": ["JavaScript", "React", "Node.js", "MongoDB", "Express", "HTML", "CSS"],
    "DevOps Engineer": ["Docker", "Kubernetes", "AWS", "Linux", "CI/CD", "Azure"],
    "Cloud Engineer": ["AWS", "Azure", "Docker", "Linux", "Google Cloud", "Kubernetes"],
    "Database Administrator": ["SQL", "MySQL", "PostgreSQL", "MongoDB", "Database Design"],
    "Business Analyst": ["Excel", "SQL", "Power BI", "Communication", "Data Analysis", "Tableau"],
    "QA Engineer": ["Selenium", "Manual Testing", "Agile", "Git", "Automation", "JIRA"],
    "Project Manager": ["Agile", "Communication", "Leadership", "Project Planning", "JIRA", "Scrum"],
    "Cyber Security Analyst": ["Network Security", "Ethical Hacking", "Linux", "Risk Assessment", "Firewalls"]
}

# ============================================
# ROUTES
# ============================================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/graphs/<path:filename>")
def serve_graph(filename):
    return send_from_directory("static/graphs", filename)

@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    try:
        if "resume" not in request.files:
            return jsonify({"status": False, "message": "No file uploaded"})
        
        file = request.files["resume"]
        if file.filename == "":
            return jsonify({"status": False, "message": "No file selected"})
        
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        
        parser = ResumeParser(filepath)
        resume_text = parser.get_resume_text()
        
        extractor = SkillExtractor()
        detected_skills = extractor.extract_skills(resume_text)
        
        return jsonify({
            "status": True,
            "resume_text": resume_text[:3000],
            "detected_skills": detected_skills
        })
    
    except Exception as e:
        return jsonify({"status": False, "message": str(e)})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not MODELS_READY:
            return jsonify({
                "status": False,
                "message": "Models not loaded. Please run 'python train_model.py' first."
            })
        
        data = request.get_json()
        
        skills = data.get("skills", "")
        education_str = data.get("education", "Bachelor")
        education = EDU_MAP.get(education_str, 2)
        experience = float(data.get("experience", 0) or 0)
        projects = float(data.get("projects", 0) or 0)
        internships = float(data.get("internships", 0) or 0)
        cgpa = float(data.get("cgpa", 0) or 0)
        certification = float(data.get("certification", 0) or 0)
        
        # Transform features
        skills_vector = tfidf.transform([skills.lower()])
        numerical = np.array([[education, experience, projects, internships, cgpa, certification]])
        numerical = scaler.transform(numerical)
        
        features = hstack([skills_vector, numerical])
        features = features.toarray()
        features = pca.transform(features)
        
        # Predict
        probabilities = model.predict_proba(features)[0]
        prediction = np.argmax(probabilities)
        predicted_role = role_encoder.inverse_transform([prediction])[0]
        confidence = round(np.max(probabilities) * 100, 2)
        
        # Top 3 roles
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_roles = []
        for idx in top_indices:
            role_name = role_encoder.inverse_transform([idx])[0]
            top_roles.append({
                "role": role_name,
                "confidence": round(probabilities[idx] * 100, 2)
            })
        
        # Calculate scores
        resume_score = calculate_resume_score(skills, education, experience, projects, internships, cgpa, certification, confidence)
        ats_score = calculate_ats_score(skills, resume_score)
        strength_analysis = analyze_strength(skills, education, experience, projects, internships, cgpa, certification, confidence)
        skill_gaps = detect_skill_gaps(skills, predicted_role)
        recommended_skills = get_recommended_skills(predicted_role, skills)
        
        return jsonify({
            "status": True,
            "predicted_role": predicted_role,
            "confidence": confidence,
            "resume_score": resume_score,
            "ats_score": ats_score,
            "top_roles": top_roles,
            "strength_analysis": strength_analysis,
            "skill_gaps": skill_gaps,
            "recommended_skills": recommended_skills
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": False, "message": str(e)})

@app.route("/health")
def health():
    return jsonify({"status": "running", "models_ready": MODELS_READY})

# ============================================
# HELPER FUNCTIONS
# ============================================

def calculate_resume_score(skills, education, experience, projects, internships, cgpa, certification, confidence):
    score = 0
    
    skill_list = [s.strip() for s in skills.split(',') if s.strip()]
    skill_count = len(skill_list)
    
    if skill_count >= 10: score += 30
    elif skill_count >= 7: score += 25
    elif skill_count >= 5: score += 20
    elif skill_count >= 3: score += 15
    else: score += skill_count * 5
    
    if education >= 3: score += 15
    elif education == 2: score += 12
    elif education == 1: score += 8
    else: score += 5
    
    if experience >= 5: score += 20
    elif experience >= 3: score += 15
    elif experience >= 1: score += 10
    else: score += 5
    
    if projects >= 5: score += 15
    elif projects >= 3: score += 10
    elif projects >= 1: score += 5
    
    if internships >= 3: score += 10
    elif internships >= 1: score += 5
    
    if cgpa >= 3.5: score += 10
    elif cgpa >= 3.0: score += 7
    elif cgpa >= 2.5: score += 4
    
    if certification == 1: score += 5
    score += min(5, confidence / 20)
    
    return min(100, int(score))

def calculate_ats_score(skills, resume_score):
    score = resume_score
    if skills and len(skills.split(',')) >= 5:
        score += 5
    
    ats_keywords = ["python", "sql", "excel", "machine learning", "data", "analytics", "project", "management", "aws", "docker"]
    skill_lower = skills.lower()
    keyword_count = sum(1 for kw in ats_keywords if kw in skill_lower)
    score += min(10, keyword_count * 2)
    
    return min(100, score)

def analyze_strength(skills, education, experience, projects, internships, cgpa, certification, confidence):
    strengths = []
    weaknesses = []
    
    skill_list = [s.strip() for s in skills.split(',') if s.strip()]
    skill_count = len(skill_list)
    
    if skill_count >= 8:
        strengths.append(f"✅ Strong technical skillset ({skill_count} skills)")
    elif skill_count < 3:
        weaknesses.append("⚠️ Add more technical skills (minimum 5 recommended)")
    
    if education >= 3:
        strengths.append("✅ Advanced degree holder")
    elif education == 0:
        weaknesses.append("⚠️ Consider higher education for better opportunities")
    
    if experience >= 4:
        strengths.append(f"✅ {int(experience)} years of relevant experience")
    elif experience == 0:
        weaknesses.append("⚠️ No work experience - consider internships")
    
    if projects >= 4:
        strengths.append(f"✅ {int(projects)} projects in portfolio")
    elif projects < 2:
        weaknesses.append("⚠️ Add more projects to showcase skills")
    
    if internships >= 2:
        strengths.append("✅ Industry internship experience")
    
    if cgpa >= 3.2:
        strengths.append(f"✅ Good academic record (CGPA: {cgpa})")
    elif cgpa < 2.5 and cgpa > 0:
        weaknesses.append("⚠️ Low CGPA - highlight other strengths")
    
    if certification == 1:
        strengths.append("✅ Professional certifications")
    else:
        weaknesses.append("⚠️ Add certifications to boost profile")
    
    if confidence >= 80:
        strengths.append("✅ Strong alignment with predicted role")
    elif confidence < 50:
        weaknesses.append("⚠️ Role mismatch - consider skill development")
    
    if len(strengths) >= 4:
        overall = "Strong"
    elif len(strengths) >= 2:
        overall = "Average"
    else:
        overall = "Needs Improvement"
    
    return {
        "overall": overall,
        "strengths": strengths[:5],
        "weaknesses": weaknesses[:5],
        "score": calculate_resume_score(skills, education, experience, projects, internships, cgpa, certification, confidence)
    }

def detect_skill_gaps(current_skills, target_role):
    current = [s.strip().lower() for s in current_skills.split(',') if s.strip()]
    required = [s.lower() for s in ROLE_SKILLS.get(target_role, [])]
    missing = [skill for skill in required if skill.lower() not in [c.lower() for c in current]]
    return missing[:8]

def get_recommended_skills(target_role, current_skills):
    current = [s.strip().lower() for s in current_skills.split(',') if s.strip()]
    recommended = []
    for skill in ROLE_SKILLS.get(target_role, []):
        if skill.lower() not in [c.lower() for c in current]:
            recommended.append(skill)
    return recommended[:10]

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 AI RESUME ANALYZER")
    print("="*50)
    if MODELS_READY:
        print("✅ Models loaded successfully!")
    else:
        print("⚠️ Models not loaded. Run 'python train_model.py' first!")
    print("🌐 Server: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(debug=True)