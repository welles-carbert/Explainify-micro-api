import os
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Explainify API", version="0.2.0")

# Allowed complexity levels
VALID_LEVELS = {"beginner", "intermediate", "advanced"}

# Request + Response Models
class ExplainRequest(BaseModel):
    text: str
    level: str = "intermediate"

class ExplainResponse(BaseModel):
    level: str
    summary: str
    explanation: str
    key_points: List[str]


# Internal API Key Verification
def verify_api_key(x_api_key: Optional[str]):
    if INTERNAL_API_KEY is None:
        return  # dev mode
    if x_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest, x_api_key: Optional[str] = Header(default=None)):

    # Validate key
    verify_api_key(x_api_key)

    # Validate text
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    # Validate level
    level = req.level.lower()
    if level not in VALID_LEVELS:
        raise HTTPException(
            status_code=400,
            detail="Invalid level. Choose: beginner, intermediate, or advanced."
        )

    # Level instructions for LLM
    level_instructions = {
        "beginner": (
            "Explain this as if to a smart 12-year-old. "
            "Use simple language and short sentences."
        ),
        "intermediate": (
            "Explain this to a college student. "
            "Be structured, clear, and reduce jargon."
        ),
        "advanced": (
            "Explain this to someone with strong domain familiarity. "
            "Use technical detail but stay concise."
        ),
    }

    system_prompt = (
        "You are an expert explanation engine.\n"
        "Follow this structure strictly:\n\n"
        "SUMMARY:\n"
        "A 2–3 sentence summary.\n\n"
        "EXPLANATION:\n"
        "A deeper explanation adapted to the requested complexity level.\n\n"
        "KEY POINTS:\n"
        "- bullet 1\n"
        "- bullet 2\n"
        "- bullet 3\n\n"
        "Be accurate and avoid hallucinations."
    )

    user_prompt = (
        f"{level_instructions[level]}\n\n"
        f"Text:\n{req.text}\n\n"
        "Return EXACTLY using the structure provided."
    )

    # Call LLM
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",   # changed from gpt-5.1-mini
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")

    content = completion.choices[0].message.content or ""

    # Clean and parse
    content = content.replace("\r", "")
    sections = {"summary": "", "explanation": "", "keypoints": []}
    current = None

    for line in content.split("\n"):
        l = line.strip()

        if l.upper().startswith("SUMMARY"):
            current = "summary"
            continue
        if l.upper().startswith("EXPLANATION"):
            current = "explanation"
            continue
        if l.upper().startswith("KEY POINTS"):
            current = "keypoints"
            continue

        if current == "summary":
            if l:
                sections["summary"] += l + " "
        elif current == "explanation":
            if l:
                sections["explanation"] += l + " "
        elif current == "keypoints":
            if l.startswith("-"):
                sections["keypoints"].append(l[1:].strip())

    # Safe fallbacks
    if not sections["summary"].strip():
        sections["summary"] = "Summary unavailable."

    if not sections["explanation"].strip():
        sections["explanation"] = "Explanation unavailable."

    if not sections["keypoints"]:
        sections["keypoints"] = ["No key points available."]

    # Final structured response
    return ExplainResponse(
        level=level,
        summary=sections["summary"].strip(),
        explanation=sections["explanation"].strip(),
        key_points=sections["keypoints"],
    )

