"""
perception/gemini_classifier.py

Shared Gemini Robotics ER 2 classification module.

WHY THIS FILE EXISTS
---------------------
Both the Offline Reasoning Validation tools AND the deployed agent need
to ask Gemini "what waste objects are in this image, and what are their
categories/locations?" - and they need to ask it in EXACTLY the same
way. If validation used a slightly different prompt or parsing routine
than the real agent, our accuracy numbers wouldn't actually reflect how
the real agent performs.

So this module is the ONE place that:
  1. Sends an image (+ a chosen prompt) to Gemini Robotics ER 2
  2. Parses the JSON response into plain Python objects
  3. Raises a clear error if the response isn't usable

Everything else (the live trial tool, run_accuracy_test.py, and later
the real agent) should import classify_image() from here instead of
calling the Gemini API directly.
"""

import os
import json
import logging

from dotenv import load_dotenv
from google import genai
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

from prompts.prompts_v1 import PROMPT_3_BOUNDING_BOXES

logger = logging.getLogger("gemini_classifier")

# ---------- Client setup (runs once, when this module is first imported) ----------
load_dotenv()
_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError("GEMINI_API_KEY not found. Check your .env file.")

_client = genai.Client(api_key=_api_key)
MODEL_NAME = "gemini-robotics-er-2-preview"


class ClassificationError(Exception):
    """Raised when Gemini's reply can't be turned into usable detections."""


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=20), reraise=True)
def _call_gemini(contents):
    logger.info("Calling Gemini Robotics ER 2...")
    return _client.models.generate_content(model=MODEL_NAME, contents=contents)


def _strip_json_fence(text: str) -> str:
    """Gemini is told to return raw JSON, but strip ```json fences defensively."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    return cleaned.strip()


def classify_image(pil_image: Image.Image, prompt: str = PROMPT_3_BOUNDING_BOXES) -> list[dict]:
    """
    Sends ONE image to Gemini Robotics ER 2 and returns a list of detections.

    With the default prompt (PROMPT_3_BOUNDING_BOXES), each detection looks like:
        {"box_2d": [ymin, xmin, ymax, xmax], "material": "plastic"}
    (box coordinates are normalized 0-1000, per Gemini's convention)

    Raises ClassificationError if Gemini's reply isn't valid, parseable JSON,
    so the caller can decide what to do (retry, log, skip this trial).
    """
    response = _call_gemini([prompt, pil_image])
    raw_text = response.text or ""
    cleaned = _strip_json_fence(raw_text)

    if not cleaned:
        raise ClassificationError("Gemini returned an empty response.")

    try:
        detections = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ClassificationError(
            f"Could not parse Gemini's response as JSON: {e}\nRaw text was:\n{raw_text}"
        ) from e

    if not isinstance(detections, list):
        raise ClassificationError(f"Expected a JSON list, got {type(detections).__name__}: {detections}")

    return detections


def classify_file(image_path: str, prompt: str = PROMPT_3_BOUNDING_BOXES) -> list[dict]:
    """Convenience wrapper for a file path instead of an in-memory PIL.Image
    (handy for quick tests against img/waste1.jpeg)."""
    return classify_image(Image.open(image_path), prompt=prompt)