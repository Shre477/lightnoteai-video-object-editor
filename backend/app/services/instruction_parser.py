"""Turns a free-text editing instruction into a structured intent that the
rest of the pipeline can act on.

Example:
    "Replace the Coca-Cola bottle with Pepsi"
    -> {"operation": "replace_object", "target": "coca-cola bottle", "replacement": "pepsi"}

Two strategies are supported:
  1. LLM-based (OpenAI or Gemini) - more robust, handles paraphrasing.
  2. Regex-based fallback - zero dependencies / zero cost, used automatically
     when no LLM_PROVIDER/API key is configured, or if the LLM call fails.
This dual-path design means "AI Integration" always has something real to
demo even without any API key on hand.
"""
import json
import re

from ..config import settings

SYSTEM_PROMPT = """You convert a video-editing instruction into strict JSON.
Schema: {"operation": "replace_object" | "remove_object" | "replace_text", "target": string, "replacement": string or null}
- "replace_object": swap one physical object for another, e.g. "replace the Coca-Cola bottle with Pepsi"
- "remove_object": delete an object entirely, e.g. "remove the bottle"
- "replace_text": swap visible text/label, e.g. "replace Coca-Cola with Pepsi" where these are text labels
"target" is the object/text being changed, "replacement" is what it becomes (or null for remove_object).
Return ONLY the JSON object, nothing else, no markdown fences."""


def _strip_code_fence(text: str) -> str:
    return re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()


def _fallback_parse(prompt: str) -> dict:
    """Simple, explainable regex parser used when no LLM is configured."""
    text = prompt.strip().lower()

    match = re.search(r"replace\s+(?:the\s+)?(.+?)\s+with\s+(.+)", text)
    if match:
        return {
            "operation": "replace_object",
            "target": match.group(1).strip().rstrip("."),
            "replacement": match.group(2).strip().rstrip("."),
        }

    match = re.search(r"remove\s+(?:the\s+)?(.+)", text)
    if match:
        return {
            "operation": "remove_object",
            "target": match.group(1).strip().rstrip("."),
            "replacement": None,
        }

    # Last resort: we couldn't confidently parse the structure, so treat the
    # whole prompt as the target of a replace-object operation with no known
    # replacement. The pipeline will surface this via `parsed_instruction`
    # so the user can see exactly what was understood.
    return {"operation": "replace_object", "target": text, "replacement": None}


def _parse_with_openai(prompt: str) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    content = _strip_code_fence(resp.choices[0].message.content)
    return json.loads(content)


def _parse_with_gemini(prompt: str) -> dict:
    import google.generativeai as genai

    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(f"{SYSTEM_PROMPT}\n\nInstruction: {prompt}")
    content = _strip_code_fence(resp.text)
    return json.loads(content)


def parse_instruction(prompt: str) -> dict:
    if settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        try:
            return _parse_with_openai(prompt)
        except Exception:
            pass  # fall through to heuristic parser below

    if settings.LLM_PROVIDER == "gemini" and settings.GEMINI_API_KEY:
        try:
            return _parse_with_gemini(prompt)
        except Exception:
            pass

    return _fallback_parse(prompt)
