import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Explainify API", version="0.1.0")


class ExplainRequest(BaseModel):
    text: str
    level: str = "intermediate"  # beginner | intermediate | advanced


class ExplainResponse(BaseModel):
    level: str
    summary: str
    explanation: str
    key_points: List[str]


def verify_api_key(x_api_key: Optional[str]) -> None:
    """
    Simple API key check. In prod, you'd store keys in a DB.
    """
    if INTERNAL_API_KEY is None:
        # No auth configured -> allow all (dev mode)
        return
    if x_api_key is None or x_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest, x_api_key: Optional[str] = Header(default=None)):
    """
    Take input text and explain it at the requested level.
    """
    verify_api_key(x_api_key)

    level = req.level.lower()
    if level not in {"beginner", "intermediate", "advanced"}:
        level = "intermediate"

    level_instructions = {
        "beginner": (
            "Explain this as if to a smart 12-year-old with no background. "
            "Use simple language and short sentences."
        ),
        "intermediate": (
            "Explain this to a college student who knows basics but not deep details. "
            "Be clear, structured, and avoid jargon unless you define it."
        ),
        "advanced": (
            "Explain this to someone with domain familiarity. "
            "You can use technical terms but stay concise and precise."
        ),
    }

    system_prompt = (
        "You are an explainer AI. You:\n"
        "- Read the user's text.\n"
        "- Create a 2–3 sentence summary.\n"
        "- Then a clear explanation at the requested level.\n"
        "- Then 3–7 bullet key points.\n"
        "Be accurate and avoid making up facts."
    )

    user_prompt = (
        f"{level_instructions[level]}\n\n"
        f"Text:\n{req.text}\n\n"
        "Return your answer in this format:\n"
        "SUMMARY:\n...\n\n"
        "EXPLANATION:\n...\n\n"
        "KEY POINTS:\n"
        "- point 1\n- point 2\n- point 3"
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    content = completion.choices[0].message.content or ""

    # Very rough parsing. We'll clean this up later.
    summary = ""
    explanation = ""
    key_points: List[str] = []

    section = None
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("SUMMARY"):
            section = "summary"
            continue
        if stripped.upper().startswith("EXPLANATION"):
            section = "explanation"
            continue
        if stripped.upper().startswith("KEY POINTS"):
            section = "keypoints"
            continue

        if not stripped:
            continue

        if section == "summary":
            summary += stripped + " "
        elif section == "explanation":
            explanation += stripped + " "
        elif section == "keypoints":
            if stripped.startswith("-"):
                key_points.append(stripped.lstrip("- ").strip())

    summary = summary.strip()
    explanation = explanation.strip()

    if not summary:
        summary = explanation[:200] + "..." if explanation else "No summary generated."
    if not key_points:
        key_points = ["No key points parsed."]

    return ExplainResponse(
        level=level,
        summary=summary,
        explanation=explanation,
        key_points=key_points,
    )
    