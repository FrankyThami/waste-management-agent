"""
Stage 1 deliverable script.

Confirms we can successfully connect to Gemini Robotics ER 2 with:
1. A text-only call
2. An image-based call, using a real waste-item photo
...using a retry/backoff wrapper to handle rate limits gracefully,
with every call logged.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------- 1. Logging setup ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("gemini_connectivity_test")

# ---------- 2. Load secrets from .env ----------
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found. Check your .env file.")

# ---------- 3. Create the Gemini client ----------
client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-robotics-er-2-preview"


# ---------- 4. Retry/backoff wrapper ----------
@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    reraise=True,
)
def call_gemini(contents):
    logger.info("Calling Gemini Robotics ER 2...")
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=contents,
    )
    logger.info("Call succeeded.")
    return response


# ---------- 5. Load our real test image ----------
# IMPORTANT: this must exactly match the filename you saw from `dir img`
IMAGE_PATH = Path("img/waste1.jpeg")


def load_test_image():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Couldn't find {IMAGE_PATH}. Check the filename/extension in your img/ folder."
        )
    return Image.open(IMAGE_PATH)


# ---------- 6. Run both test calls ----------
WASTE_CATEGORIES = ["Paper", "Metal", "Organic", "Unsorted", "Plastic", "Glass"]

CLASSIFICATION_PROMPT = f"""
Look at this image. List each distinct waste item you can see.
For each item, assign exactly one category from this list: {WASTE_CATEGORIES}.

Respond in this format, one line per item:
<item name> - <category>
"""


def main():
    # Test 1: text-only call
    text_response = call_gemini("In one short sentence, what is a 6-DOF robot arm?")
    print("\n--- TEXT RESPONSE ---")
    print(text_response.text)

    # Test 2: image-based call, using a real waste photo
    test_image = load_test_image()
    image_response = call_gemini([CLASSIFICATION_PROMPT, test_image])
    print("\n--- IMAGE RESPONSE (waste classification) ---")
    print(image_response.text)

    logger.info("Both text and image calls to Gemini Robotics ER 2 succeeded.")


if __name__ == "__main__":
    main()