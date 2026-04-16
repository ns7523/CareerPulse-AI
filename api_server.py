from __future__ import annotations

import importlib.util
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

PROJECT_ROOT = Path(__file__).resolve().parent


def load_symbol(relative_path: str, symbol_name: str) -> Any:
    module_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(f"careerpulse_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol_name)


JOB_ROLES = load_symbol("config/job_roles.py", "JOB_ROLES")
ResumeAnalyzer = load_symbol("utils/resume_analyzer.py", "ResumeAnalyzer")
ResumeParser = load_symbol("utils/resume_parser.py", "ResumeParser")

app = FastAPI(title="CareerPulse AI API", version="1.0.0")

resume_parser = ResumeParser()
resume_analyzer = ResumeAnalyzer()

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
CERTIFICATION_KEYWORDS = {
    "certification",
    "certifications",
    "certificate",
    "certificates",
    "license",
    "licenses",
    "credential",
    "credentials",
}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    target_category: str | None = Form(default=None),
    target_role: str | None = Form(default=None),
) -> dict[str, Any]:
    filename = file.filename or "resume"
    extension = file_extension(filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    file_obj = named_bytes_io(file_bytes, filename)
    parsed_resume = resume_parser.parse(file_obj)
    raw_text = (parsed_resume.get("raw_text") or "").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Could not extract text from the uploaded resume.")

    role_context = resolve_role_context(raw_text, target_category, target_role)
    analysis = resume_analyzer.analyze_resume(parsed_resume, role_context["job_requirements"])

    certifications = extract_certifications(raw_text)
    suggested_roles = suggest_roles(raw_text)

    return {
        "status": "success",
        "file_name": filename,
        "target": {
            "category": role_context["category"],
            "role": role_context["role"],
        },
        "extracted_data": {
            "document_type": analysis.get("document_type", "unknown"),
            "raw_text": raw_text,
            "personal_info": {
                "name": analysis.get("name", ""),
                "email": analysis.get("email", ""),
                "phone": analysis.get("phone", ""),
                "linkedin": analysis.get("linkedin", ""),
                "github": analysis.get("github", ""),
                "portfolio": analysis.get("portfolio", ""),
            },
            "summary": analysis.get("summary", ""),
            "skills": analysis.get("skills", []),
            "education": analysis.get("education", []),
            "experience": analysis.get("experience", []),
            "projects": analysis.get("projects", []),
            "certifications": certifications,
        },
        "analysis": {
            "ats_score": analysis.get("ats_score", 0),
            "section_score": analysis.get("section_score", 0),
            "format_score": analysis.get("format_score", 0),
            "keyword_match": analysis.get("keyword_match", {"score": 0, "found_skills": [], "missing_skills": []}),
            "section_scores": analysis.get("section_scores", {}),
            "suggestions": analysis.get("suggestions", []),
        },
        "suggested_roles": suggested_roles,
    }


def named_bytes_io(file_bytes: bytes, filename: str) -> BytesIO:
    file_obj = BytesIO(file_bytes)
    file_obj.name = filename
    return file_obj


def file_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def resolve_role_context(raw_text: str, target_category: str | None, target_role: str | None) -> dict[str, Any]:
    if target_role:
        for category_name, roles in JOB_ROLES.items():
            if target_category and category_name != target_category:
                continue
            if target_role in roles:
                return {
                    "category": category_name,
                    "role": target_role,
                    "job_requirements": roles[target_role],
                }

    suggested_roles = suggest_roles(raw_text)
    if suggested_roles:
        top_match = suggested_roles[0]
        job_requirements = JOB_ROLES[top_match["category"]][top_match["role"]]
        return {
            "category": top_match["category"],
            "role": top_match["role"],
            "job_requirements": job_requirements,
        }

    return {
        "category": None,
        "role": None,
        "job_requirements": {"required_skills": []},
    }


def suggest_roles(raw_text: str, limit: int = 3) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []

    for category_name, roles in JOB_ROLES.items():
        for role_name, role_data in roles.items():
            required_skills = role_data.get("required_skills", [])
            keyword_match = resume_analyzer.calculate_keyword_match(raw_text, required_skills)
            if keyword_match["score"] <= 0:
                continue
            suggestions.append(
                {
                    "category": category_name,
                    "role": role_name,
                    "score": round(keyword_match["score"], 2),
                    "found_skills": keyword_match["found_skills"],
                    "missing_skills": keyword_match["missing_skills"],
                }
            )

    suggestions.sort(key=lambda item: item["score"], reverse=True)
    return suggestions[:limit]


def extract_certifications(raw_text: str) -> list[str]:
    certifications: list[str] = []
    current_block: list[str] = []
    in_cert_section = False

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            if in_cert_section and current_block:
                certifications.append(" ".join(current_block))
                current_block = []
            continue

        lowered = line.lower()
        if any(keyword in lowered for keyword in CERTIFICATION_KEYWORDS):
            if line.lower() not in CERTIFICATION_KEYWORDS:
                current_block.append(line)
            in_cert_section = True
            continue

        if in_cert_section and looks_like_section_header(lowered):
            if current_block:
                certifications.append(" ".join(current_block))
            current_block = []
            in_cert_section = False
            continue

        if in_cert_section:
            current_block.append(line)

    if current_block:
        certifications.append(" ".join(current_block))

    return certifications


def looks_like_section_header(lowered_line: str) -> bool:
    section_keywords = (
        "education",
        "experience",
        "projects",
        "skills",
        "summary",
        "objective",
        "achievements",
        "contact",
    )
    return any(keyword in lowered_line for keyword in section_keywords)
