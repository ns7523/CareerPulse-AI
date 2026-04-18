from __future__ import annotations

import importlib.util
import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

PROJECT_ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("careerpulse.api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())


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
DEFAULT_AI_ORDER = ["gemini", "openrouter", "openai", "grok", "ollama"]
DEFAULT_PROVIDER_MODELS = {
    "gemini": [
        "models/gemini-2.5-pro",
        "models/gemini-1.5-pro",
        "models/gemini-1.5-flash",
    ],
    "openrouter": [
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "mistralai/mistral-small-3.1-24b-instruct:free",
    ],
    "claude": [
        "anthropic/claude-3.5-sonnet",
        "anthropic/claude-3-haiku",
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
    ],
    "grok": [
        "grok-2-latest",
        "grok-beta",
    ],
    "ollama": [
        "llama3.2",
        "mistral",
        "phi3",
    ],
}
PROVIDER_LABELS = {
    "gemini": "Google Gemini",
    "openrouter": "OpenRouter",
    "openai": "OpenAI",
    "grok": "Grok",
    "ollama": "Ollama",
    "claude": "Claude via OpenRouter",
}
AI_STATUS_TEXT = {
    "enhanced": "Enhanced mode enabled",
    "free": "AI is ready (Free mode)",
    "local": "Local AI fallback enabled",
    "basic": "Basic processing mode enabled",
}
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


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> str:
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>CareerPulse AI API</title>
        <style>
          :root {
            color-scheme: dark;
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }
          body {
            margin: 0;
            min-height: 100vh;
            display: grid;
            place-items: center;
            background: #0f1720;
            color: #e8eef5;
          }
          main {
            width: min(720px, calc(100vw - 32px));
            padding: 28px;
            border: 1px solid #334155;
            border-radius: 12px;
            background: #17212b;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
          }
          h1 {
            margin: 0 0 12px;
            font-size: 32px;
          }
          p {
            margin: 0 0 14px;
            line-height: 1.6;
            color: #c7d2de;
          }
          code {
            padding: 2px 6px;
            border-radius: 6px;
            background: #0f1720;
            color: #86efac;
          }
          ul {
            margin: 16px 0 0;
            padding-left: 18px;
            color: #c7d2de;
          }
          a {
            color: #7dd3fc;
            text-decoration: none;
          }
        </style>
      </head>
      <body>
        <main>
          <h1>CareerPulse AI API</h1>
          <p>The backend is live and ready for Android resume upload and analysis.</p>
          <p>Health check: <code>/health</code></p>
          <p>Upload endpoint: <code>POST /upload-resume</code></p>
          <p>Interactive docs: <a href="/docs">/docs</a></p>
          <ul>
            <li>Accepted files: PDF, DOCX</li>
            <li>Returns extracted resume data, ATS-style analysis, missing skills, and suggested roles</li>
            <li>AI analysis uses automatic provider fallback before dropping to basic mode</li>
          </ul>
        </main>
      </body>
    </html>
    """


@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    target_category: str | None = Form(default=None),
    target_role: str | None = Form(default=None),
    job_description: str | None = Form(default=None),
    provider: str | None = Form(default=None),
    ai_mode: str | None = Form(default="auto"),
    model: str | None = Form(default=None),
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
    base_analysis = resume_analyzer.analyze_resume(parsed_resume, role_context["job_requirements"])
    ai_result = run_ai_analysis_pipeline(
        resume_text=raw_text,
        role_context=role_context,
        provider=provider,
        ai_mode=ai_mode,
        requested_model=model,
        job_description=job_description,
        fallback_analysis=base_analysis,
    )
    analysis = merge_analysis(base_analysis, ai_result.get("analysis"))

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
            "resume_score": analysis.get("resume_score", analysis.get("ats_score", 0)),
            "section_score": analysis.get("section_score", 0),
            "format_score": analysis.get("format_score", 0),
            "keyword_match": analysis.get("keyword_match", {"score": 0, "found_skills": [], "missing_skills": []}),
            "section_scores": analysis.get("section_scores", {}),
            "suggestions": analysis.get("suggestions", []),
            "strengths": analysis.get("strengths", []),
            "career_paths": analysis.get("career_paths", []),
            "interview_questions": analysis.get(
                "interview_questions",
                {"technical": [], "behavioral": [], "project": []},
            ),
            "recommended_courses": analysis.get("recommended_courses", []),
            "job_match_score": analysis.get("job_match_score", 0),
        },
        "suggested_roles": suggested_roles,
        "ai_processing": ai_result["processing"],
    }


def named_bytes_io(file_bytes: bytes, filename: str) -> BytesIO:
    file_obj = BytesIO(file_bytes)
    file_obj.name = filename
    return file_obj


def file_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def run_ai_analysis_pipeline(
    resume_text: str,
    role_context: dict[str, Any],
    provider: str | None,
    ai_mode: str | None,
    requested_model: str | None,
    job_description: str | None,
    fallback_analysis: dict[str, Any],
) -> dict[str, Any]:
    attempts: list[dict[str, str]] = []
    role_name = role_context.get("role")
    required_skills = role_context.get("job_requirements", {}).get("required_skills", [])
    prompt = build_ai_prompt(
        resume_text=resume_text,
        role_name=role_name,
        required_skills=required_skills,
        job_description=job_description,
        fallback_analysis=fallback_analysis,
    )

    for provider_key, model_name in build_attempt_plan(provider, ai_mode, requested_model):
        availability_issue = provider_unavailable_reason(provider_key)
        if availability_issue:
            attempts.append(
                {
                    "provider": display_provider_name(provider_key, model_name),
                    "model": model_name,
                    "status": "skipped",
                    "reason": availability_issue,
                }
            )
            continue

        try:
            response_text = call_ai_provider(provider_key, model_name, prompt)
            structured = normalize_ai_result(parse_ai_json_response(response_text))
            attempts.append(
                {
                    "provider": display_provider_name(provider_key, model_name),
                    "model": model_name,
                    "status": "success",
                    "reason": "",
                }
            )
            return {
                "analysis": structured,
                "processing": {
                    "mode": provider_mode(provider_key, model_name),
                    "status_text": AI_STATUS_TEXT[provider_mode(provider_key, model_name)],
                    "provider_used": display_provider_name(provider_key, model_name),
                    "model_used": model_name,
                    "fallback_used": len(attempts) > 1,
                    "attempts": attempts,
                },
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("AI analysis failed for %s (%s): %s", provider_key, model_name, exc)
            attempts.append(
                {
                    "provider": display_provider_name(provider_key, model_name),
                    "model": model_name,
                    "status": "failed",
                    "reason": str(exc),
                }
            )

    return {
        "analysis": None,
        "processing": {
            "mode": "basic",
            "status_text": AI_STATUS_TEXT["basic"],
            "provider_used": "Basic Analyzer",
            "model_used": "Rule-based fallback",
            "fallback_used": True,
            "attempts": attempts,
        },
    }


def build_attempt_plan(
    requested_provider: str | None,
    ai_mode: str | None,
    requested_model: str | None,
) -> list[tuple[str, str]]:
    if (ai_mode or "").strip().lower() == "basic":
        return []

    preferred = normalize_provider(requested_provider)
    plan: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add_attempt(provider_key: str, model_name: str) -> None:
        attempt = (provider_key, model_name)
        if attempt not in seen:
            seen.add(attempt)
            plan.append(attempt)

    if preferred:
        for model_name in provider_models(preferred, requested_model):
            add_attempt(base_provider(preferred), model_name)

    if requested_model and not preferred:
        for provider_key in DEFAULT_AI_ORDER:
            add_attempt(provider_key, requested_model)

    for provider_key in DEFAULT_AI_ORDER:
        for model_name in provider_models(provider_key, None):
            add_attempt(provider_key, model_name)

    return plan


def provider_models(provider_key: str, requested_model: str | None) -> list[str]:
    models = list(DEFAULT_PROVIDER_MODELS.get(provider_key, []))
    if requested_model:
        models.insert(0, requested_model)
    return list(dict.fromkeys(models))


def normalize_provider(provider: str | None) -> str | None:
    if not provider:
        return None
    normalized = provider.strip().lower()
    aliases = {
        "google": "gemini",
        "gemini": "gemini",
        "openrouter": "openrouter",
        "claude": "claude",
        "anthropic": "claude",
        "openai": "openai",
        "grok": "grok",
        "xai": "grok",
        "ollama": "ollama",
        "local": "ollama",
        "custom": None,
    }
    return aliases.get(normalized, normalized if normalized in DEFAULT_PROVIDER_MODELS else None)


def base_provider(provider_key: str) -> str:
    if provider_key == "claude":
        return "openrouter"
    return provider_key


def provider_mode(provider_key: str, model_name: str) -> str:
    if provider_key == "gemini":
        return "free"
    if provider_key == "ollama":
        return "local"
    if provider_key == "openrouter" and ":free" in model_name:
        return "free"
    return "enhanced"


def display_provider_name(provider_key: str, model_name: str) -> str:
    if provider_key == "openrouter" and model_name.startswith("anthropic/claude"):
        return PROVIDER_LABELS["claude"]
    return PROVIDER_LABELS.get(provider_key, provider_key.title())


def provider_unavailable_reason(provider_key: str) -> str | None:
    if provider_key == "gemini" and not os.getenv("GOOGLE_API_KEY"):
        return "GOOGLE_API_KEY is not configured"
    if provider_key == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
        return "OPENROUTER_API_KEY is not configured"
    if provider_key == "openai" and not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is not configured"
    if provider_key == "grok" and not (os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")):
        return "XAI_API_KEY or GROK_API_KEY is not configured"
    return None


def build_ai_prompt(
    resume_text: str,
    role_name: str | None,
    required_skills: list[str],
    job_description: str | None,
    fallback_analysis: dict[str, Any],
) -> str:
    required_skills_text = ", ".join(required_skills) if required_skills else "Not specified"
    fallback_missing = ", ".join(fallback_analysis.get("keyword_match", {}).get("missing_skills", [])) or "None detected"
    return f"""
You are an expert resume analyst. Review the resume and return only valid JSON.

Return this exact JSON shape:
{{
  "summary": "2-4 sentence summary",
  "skills": {{
    "current": ["skill"],
    "missing": ["skill"]
  }},
  "strengths": ["strength"],
  "suggestions": ["actionable suggestion"],
  "career_paths": ["recommended role"],
  "recommended_courses": ["course or certification"],
  "interview_questions": {{
    "technical": ["question"],
    "behavioral": ["question"],
    "project": ["question"]
  }},
  "ats_score": 0,
  "resume_score": 0,
  "job_match_score": 0
}}

Rules:
- Output valid JSON only, with no markdown fences.
- Keep every list concise and useful.
- Keep scores as integers from 0 to 100.
- Use the target role, required skills, and job description if provided.

Target role: {role_name or "Not specified"}
Required skills: {required_skills_text}
Baseline missing skills from ATS scan: {fallback_missing}
Job description: {job_description or "Not provided"}

Resume:
{resume_text}
""".strip()


def call_ai_provider(provider_key: str, model_name: str, prompt: str) -> str:
    if provider_key == "gemini":
        return call_gemini(model_name, prompt)
    if provider_key == "openrouter":
        return call_openai_compatible(
            url="https://openrouter.ai/api/v1/chat/completions",
            api_key=os.getenv("OPENROUTER_API_KEY", ""),
            model_name=model_name,
            prompt=prompt,
            extra_headers={
                "HTTP-Referer": "https://careerpulse-ai.local",
                "X-Title": "CareerPulse AI",
            },
        )
    if provider_key == "openai":
        return call_openai_compatible(
            url="https://api.openai.com/v1/chat/completions",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model_name=model_name,
            prompt=prompt,
        )
    if provider_key == "grok":
        return call_openai_compatible(
            url="https://api.x.ai/v1/chat/completions",
            api_key=os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY", ""),
            model_name=model_name,
            prompt=prompt,
        )
    if provider_key == "ollama":
        return call_ollama(model_name, prompt)
    raise ValueError(f"Unsupported AI provider: {provider_key}")


def call_gemini(model_name: str, prompt: str) -> str:
    try:
        import google.generativeai as genai
    except ImportError as exc:  # pragma: no cover - dependency-level issue
        raise RuntimeError("google-generativeai is not installed") from exc

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not configured")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(prompt)
    text = getattr(response, "text", "") or ""
    if not text.strip():
        raise RuntimeError("Gemini returned an empty response")
    return text


def call_openai_compatible(
    url: str,
    api_key: str,
    model_name: str,
    prompt: str,
    extra_headers: dict[str, str] | None = None,
) -> str:
    if not api_key:
        raise RuntimeError("API key is not configured")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    response = requests.post(
        url,
        headers=headers,
        json={
            "model": model_name,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise resume analysis engine. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
        },
        timeout=45,
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("No completion choices were returned")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("The AI provider returned an empty completion")
    return content


def call_ollama(model_name: str, prompt: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model_name,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise resume analysis engine. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    content = payload.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("Ollama returned an empty response")
    return content


def parse_ai_json_response(response_text: str) -> dict[str, Any]:
    text = (response_text or "").strip()
    if not text:
        raise ValueError("The AI response was empty")

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    while start != -1:
        candidate = extract_first_json_object(text[start:])
        if candidate:
            return json.loads(candidate)
        start = text.find("{", start + 1)

    raise ValueError("Could not parse JSON from the AI response")


def extract_first_json_object(text: str) -> str | None:
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if char == '"' and not escape:
            in_string = not in_string
        elif char == "\\" and in_string:
            escape = not escape
            continue

        if not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[: index + 1]

        if char != "\\":
            escape = False

    return None


def normalize_ai_result(data: dict[str, Any]) -> dict[str, Any]:
    skills = data.get("skills") if isinstance(data.get("skills"), dict) else {}
    interview_questions = data.get("interview_questions") if isinstance(data.get("interview_questions"), dict) else {}

    return {
        "summary": safe_string(data.get("summary")),
        "skills": unique_strings(skills.get("current")),
        "missing_skills": unique_strings(skills.get("missing")),
        "strengths": unique_strings(data.get("strengths")),
        "suggestions": unique_strings(data.get("suggestions")),
        "career_paths": unique_strings(data.get("career_paths")),
        "recommended_courses": unique_strings(data.get("recommended_courses")),
        "interview_questions": {
            "technical": unique_strings(interview_questions.get("technical")),
            "behavioral": unique_strings(interview_questions.get("behavioral")),
            "project": unique_strings(interview_questions.get("project")),
        },
        "ats_score": safe_score(data.get("ats_score")),
        "resume_score": safe_score(data.get("resume_score")),
        "job_match_score": safe_score(data.get("job_match_score")),
    }


def merge_analysis(base_analysis: dict[str, Any], ai_analysis: dict[str, Any] | None) -> dict[str, Any]:
    if not ai_analysis:
        base_analysis["resume_score"] = base_analysis.get("ats_score", 0)
        base_analysis["strengths"] = []
        base_analysis["career_paths"] = []
        base_analysis["recommended_courses"] = []
        base_analysis["interview_questions"] = {"technical": [], "behavioral": [], "project": []}
        base_analysis["job_match_score"] = 0
        return base_analysis

    suggestions = unique_strings([*(base_analysis.get("suggestions", []) or []), *ai_analysis.get("suggestions", [])])
    found_skills = unique_strings([*(base_analysis.get("skills", []) or []), *ai_analysis.get("skills", [])])
    missing_skills = unique_strings(
        [
            *(base_analysis.get("keyword_match", {}).get("missing_skills", []) or []),
            *ai_analysis.get("missing_skills", []),
        ]
    )

    merged = {
        **base_analysis,
        "summary": ai_analysis.get("summary") or base_analysis.get("summary", ""),
        "skills": found_skills or base_analysis.get("skills", []),
        "resume_score": ai_analysis.get("resume_score") or base_analysis.get("ats_score", 0),
        "ats_score": ai_analysis.get("ats_score") or base_analysis.get("ats_score", 0),
        "suggestions": suggestions,
        "strengths": ai_analysis.get("strengths", []),
        "career_paths": ai_analysis.get("career_paths", []),
        "recommended_courses": ai_analysis.get("recommended_courses", []),
        "interview_questions": ai_analysis.get(
            "interview_questions",
            {"technical": [], "behavioral": [], "project": []},
        ),
        "job_match_score": ai_analysis.get("job_match_score", 0),
    }

    keyword_match = dict(base_analysis.get("keyword_match", {}))
    keyword_match["found_skills"] = found_skills
    keyword_match["missing_skills"] = missing_skills
    keyword_match["score"] = calculate_keyword_score(found_skills, missing_skills)
    merged["keyword_match"] = keyword_match
    return merged


def calculate_keyword_score(found_skills: list[str], missing_skills: list[str]) -> float:
    total = len(found_skills) + len(missing_skills)
    if total <= 0:
        return 0.0
    return round((len(found_skills) / total) * 100, 2)


def safe_score(value: Any) -> int:
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(0, min(numeric, 100))


def unique_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = safe_string(value)
        if text and text not in seen:
            seen.add(text)
            cleaned.append(text)
    return cleaned


def safe_string(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


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
