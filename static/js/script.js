let chartInstance = null;

document.addEventListener("DOMContentLoaded", function() {
    initializeDragDrop();
    initializeFileInput();
});

function initializeDragDrop() {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("resumeFile");
    
    if (!dropZone) return;
    
    dropZone.addEventListener("click", () => fileInput.click());
    
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });
    
    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });
    
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            uploadResume();
        }
    });
}

function initializeFileInput() {
    const fileInput = document.getElementById("resumeFile");
    if (fileInput) {
        fileInput.addEventListener("change", () => {
            if (fileInput.files.length > 0) uploadResume();
        });
    }
}

async function uploadResume() {
    const file = document.getElementById("resumeFile").files[0];
    
    if (!file) {
        showToast("Please select a PDF file", "error");
        return;
    }
    
    if (file.type !== "application/pdf") {
        showToast("Only PDF files are allowed", "error");
        return;
    }
    
    showLoading();
    
    const formData = new FormData();
    formData.append("resume", file);
    
    try {
        const response = await fetch("/upload_resume", {
            method: "POST",
            body: formData
        });
        
        const result = await response.json();
        hideLoading();
        
        if (result.status) {
            document.getElementById("extractedText").value = result.resume_text;
            displayDetectedSkills(result.detected_skills);
            
            if (result.detected_skills && result.detected_skills.length > 0) {
                document.getElementById("skills").value = result.detected_skills.join(", ");
            }
            
            showToast("Resume analyzed successfully!", "success");
        } else {
            showToast(result.message || "Error processing resume", "error");
        }
    } catch (error) {
        hideLoading();
        showToast("Network error. Please try again.", "error");
        console.error(error);
    }
}

function displayDetectedSkills(skills) {
    const container = document.getElementById("detectedSkills");
    
    if (!container) return;
    
    if (!skills || skills.length === 0) {
        container.innerHTML = '<p class="text-muted">No skills detected. Try uploading a resume with more text.</p>';
        return;
    }
    
    container.innerHTML = skills.map(skill => 
        `<span class="skill-badge"><i class="fas fa-code"></i> ${escapeHtml(skill)}</span>`
    ).join('');
}

async function predictRole() {
    const skills = document.getElementById("skills").value;
    const education = document.getElementById("education").value;
    const experience = document.getElementById("experience").value;
    const projects = document.getElementById("projects").value;
    const internships = document.getElementById("internships").value;
    const cgpa = document.getElementById("cgpa").value;
    const certification = document.getElementById("certification").value;
    
    if (!skills || skills.trim() === "") {
        showToast("Please enter skills or upload a resume first", "error");
        return;
    }
    
    showLoading();
    
    const requestData = {
        skills: skills,
        education: education,
        experience: experience || 0,
        projects: projects || 0,
        internships: internships || 0,
        cgpa: cgpa || 0,
        certification: certification || 0
    };
    
    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(requestData)
        });
        
        const result = await response.json();
        hideLoading();
        
        if (result.status) {
            displayResults(result);
            showToast("Analysis completed successfully!", "success");
        } else {
            showToast(result.message || "Prediction error", "error");
        }
    } catch (error) {
        hideLoading();
        showToast("Network error. Please try again.", "error");
        console.error(error);
    }
}

function displayResults(result) {
    const resultSection = document.getElementById("resultSection");
    if (resultSection) {
        resultSection.style.display = "block";
        resultSection.scrollIntoView({ behavior: "smooth" });
    }
    
    document.getElementById("role").innerHTML = escapeHtml(result.predicted_role || "N/A");
    document.getElementById("confidence").innerHTML = `${result.confidence || 0}<small style="font-size:14px;">%</small>`;
    document.getElementById("resumeScore").innerHTML = `${result.resume_score || 0}<small style="font-size:14px;">/100</small>`;
    document.getElementById("atsScore").innerHTML = `${result.ats_score || 0}<small style="font-size:14px;">/100</small>`;
    
    if (result.top_roles && result.top_roles.length > 0) {
        const topRolesHtml = result.top_roles.map(role => 
            `<div class="role-badge">
                <i class="fas fa-briefcase"></i> ${escapeHtml(role.role)}
                <span class="badge bg-light text-dark ms-2">${role.confidence}%</span>
            </div>`
        ).join('');
        document.getElementById("topRoles").innerHTML = topRolesHtml;
    }
    
    if (result.strength_analysis) {
        const sa = result.strength_analysis;
        const alertClass = sa.overall === "Strong" ? "success" : sa.overall === "Average" ? "warning" : "danger";
        
        document.getElementById("strengthAnalysis").innerHTML = `
            <div class="alert alert-${alertClass} mb-3">
                <strong>Overall: ${sa.overall}</strong> | Score: ${sa.score}/100
            </div>
            <div class="row">
                <div class="col-md-6">
                    <h6 class="text-success"><i class="fas fa-check-circle"></i> Strengths</h6>
                    <ul>${sa.strengths.map(s => `<li>${escapeHtml(s)}</li>`).join('') || '<li>None identified</li>'}</ul>
                </div>
                <div class="col-md-6">
                    <h6 class="text-warning"><i class="fas fa-exclamation-circle"></i> Areas to Improve</h6>
                    <ul>${sa.weaknesses.map(w => `<li>${escapeHtml(w)}</li>`).join('') || '<li>None identified</li>'}</ul>
                </div>
            </div>
        `;
    }
    
    if (result.skill_gaps && result.skill_gaps.length > 0) {
        document.getElementById("skillGaps").innerHTML = result.skill_gaps.map(skill => 
            `<span class="missing-skill"><i class="fas fa-times-circle"></i> ${escapeHtml(skill)}</span>`
        ).join('');
    } else {
        document.getElementById("skillGaps").innerHTML = '<p class="text-success">No major skill gaps detected!</p>';
    }
    
    if (result.recommended_skills && result.recommended_skills.length > 0) {
        document.getElementById("recommendedSkills").innerHTML = result.recommended_skills.map(skill => 
            `<span class="skill-badge"><i class="fas fa-plus-circle"></i> ${escapeHtml(skill)}</span>`
        ).join('');
    }
    
    if (result.top_roles && result.top_roles.length > 0) {
        createConfidenceChart(result.top_roles);
    }
}

function createConfidenceChart(topRoles) {
    const canvas = document.getElementById("roleChart");
    if (!canvas) return;
    
    if (chartInstance) chartInstance.destroy();
    
    chartInstance = new Chart(canvas, {
        type: "bar",
        data: {
            labels: topRoles.map(r => r.role),
            datasets: [{
                label: "Confidence Score (%)",
                data: topRoles.map(r => r.confidence),
                backgroundColor: ["#667eea", "#764ba2", "#f093fb"],
                borderRadius: 10,
                barPercentage: 0.6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { position: "top" },
                tooltip: { callbacks: { label: (ctx) => `Confidence: ${ctx.raw}%` } }
            },
            scales: {
                y: { 
                    beginAtZero: true, 
                    max: 100, 
                    title: { display: true, text: "Confidence (%)" },
                    grid: { dashArray: [5, 5] }
                },
                x: { title: { display: true, text: "Job Roles" } }
            }
        }
    });
}

function showLoading() {
    let loader = document.getElementById("loadingOverlay");
    if (!loader) {
        loader = document.createElement("div");
        loader.id = "loadingOverlay";
        loader.className = "loading-overlay";
        loader.innerHTML = '<div class="spinner-custom"></div><p style="color:white; margin-top:20px;">Processing...</p>';
        document.body.appendChild(loader);
    }
    loader.style.display = "flex";
}

function hideLoading() {
    const loader = document.getElementById("loadingOverlay");
    if (loader) loader.style.display = "none";
}

function showToast(message, type) {
    let toastContainer = document.getElementById("toastContainer");
    if (!toastContainer) {
        toastContainer = document.createElement("div");
        toastContainer.id = "toastContainer";
        toastContainer.style.position = "fixed";
        toastContainer.style.bottom = "20px";
        toastContainer.style.right = "20px";
        toastContainer.style.zIndex = "9999";
        document.body.appendChild(toastContainer);
    }
    
    const toast = document.createElement("div");
    toast.className = "toast-notification";
    toast.style.background = type === "success" ? "#10b981" : "#ef4444";
    toast.style.color = "white";
    toast.innerHTML = `<i class="fas ${type === "success" ? "fa-check-circle" : "fa-exclamation-circle"}"></i> ${escapeHtml(message)}`;
    
    toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateX(100%)";
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function toggleDarkMode() {
    document.body.classList.toggle("dark-mode");
    const icon = document.querySelector(".btn-outline-light i");
    if (icon) {
        icon.className = document.body.classList.contains("dark-mode") ? "fas fa-sun" : "fas fa-moon";
    }
    
    if (chartInstance) {
        const topRoles = getCurrentTopRoles();
        if (topRoles.length) createConfidenceChart(topRoles);
    }
}

function getCurrentTopRoles() {
    const roleElements = document.querySelectorAll("#topRoles .role-badge");
    const roles = [];
    roleElements.forEach(el => {
        const text = el.innerText;
        const match = text.match(/(.+?)\s+(\d+)%/);
        if (match) {
            roles.push({
                role: match[1].trim(),
                confidence: parseInt(match[2])
            });
        }
    });
    return roles;
}

function scrollToResults() {
    const resultSection = document.getElementById("resultSection");
    if (resultSection && resultSection.style.display === "block") {
        resultSection.scrollIntoView({ behavior: "smooth" });
    }
}

function escapeHtml(str) {
    if (!str) return "";
    return str
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}