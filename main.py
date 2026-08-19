import os
import io
import time
import tempfile
import logging
from typing import Optional

import requests
from PIL import Image, UnidentifiedImageError

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from deep_translator import GoogleTranslator
from huggingface_hub import InferenceClient


# =========================================================
# CONFIG
# =========================================================

APP_NAME = "VIP AI Photo Editor API"

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# Main image editing model
MODEL = os.getenv(
    "IMAGE_EDIT_MODEL",
    "Qwen/Qwen-Image-Edit"
)

# Provider can be changed from Render Environment Variables.
# Example: fal-ai / auto
HF_PROVIDER = os.getenv("HF_PROVIDER", "auto").strip()

MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "15"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(APP_NAME)


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title=APP_NAME,
    version="2.0.0",
    description="Natural-language AI Photo Editing API"
)


# =========================================================
# REQUEST MODEL
# =========================================================

class EditRequest(BaseModel):
    image_url: str = Field(
        ...,
        min_length=5,
        description="Public URL of the input image"
    )

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural-language editing instruction"
    )


# =========================================================
# RESPONSE HELPERS
# =========================================================

def success_response(
    photo_url: str,
    original_prompt: str,
    translated_prompt: str
):
    return {
        "status": "success",
        "photo_url": photo_url,
        "prompt": original_prompt,
        "translated_prompt": translated_prompt,
        "model": MODEL
    }


def error_response(message: str):
    return {
        "status": "error",
        "error": message
    }


# =========================================================
# PROMPT TRANSLATION
# =========================================================

def translate_prompt(prompt: str) -> str:

    prompt = prompt.strip()

    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    try:
        translated = GoogleTranslator(
            source="auto",
            target="en"
        ).translate(prompt)

        if translated and translated.strip():
            return translated.strip()

    except Exception as e:
        logger.warning(
            "Translation failed, using original prompt: %s",
            e
        )

    # If translation service fails, don't destroy the request.
    return prompt


# =========================================================
# IMAGE DOWNLOAD
# =========================================================

def download_image(image_url: str) -> bytes:

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(compatible; VIP-AI-Photo-Editor/2.0)"
        )
    }

    response = requests.get(
        image_url,
        headers=headers,
        timeout=30,
        allow_redirects=True
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "content-type",
        ""
    ).lower()

    data = response.content

    if len(data) > MAX_IMAGE_MB * 1024 * 1024:
        raise ValueError(
            f"Image is larger than {MAX_IMAGE_MB} MB."
        )

    # Validate that the downloaded content is actually an image.
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
    except (UnidentifiedImageError, Exception) as e:
        raise ValueError(
            "The provided URL does not contain a valid image."
        ) from e

    if (
        not content_type.startswith("image/")
        and not data.startswith(b"\x89PNG")
        and not data.startswith(b"\xff\xd8")
        and not data.startswith(b"RIFF")
    ):
        logger.warning(
            "Image URL returned unusual content-type: %s",
            content_type
        )

    return data


# =========================================================
# IMAGE NORMALIZATION
# =========================================================

def normalize_image(image_bytes: bytes) -> bytes:

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    # Remove problematic modes.
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA")

    # Preserve alpha when present.
    if image.mode == "RGBA":
        background = Image.new(
            "RGB",
            image.size,
            "white"
        )

        background.paste(
            image,
            mask=image.getchannel("A")
        )

        image = background
    else:
        image = image.convert("RGB")

    # Prevent extremely huge images from causing
    # unnecessary memory usage.
    max_dimension = 2048

    if max(image.size) > max_dimension:

        ratio = max_dimension / max(image.size)

        new_size = (
            max(1, int(image.width * ratio)),
            max(1, int(image.height * ratio))
        )

        image = image.resize(
            new_size,
            Image.Resampling.LANCZOS
        )

    output = io.BytesIO()

    image.save(
        output,
        format="PNG",
        optimize=True
    )

    return output.getvalue()


# =========================================================
# PROMPT BUILDER
# =========================================================

def build_edit_prompt(prompt: str) -> str:

    base_instruction = """
Edit the provided image according to the user's instruction.

IMPORTANT:
- Preserve the identity of the main person whenever possible.
- Preserve facial structure unless the user explicitly asks to change it.
- Preserve natural anatomy.
- Preserve realistic proportions.
- Do not randomly add people.
- Do not randomly remove important objects.
- Change only what the user's instruction requests.
- Keep unrelated areas visually consistent.
- Produce a natural, high-quality result.
- If the instruction asks for a background change, keep the main
  subject consistent while changing the environment.
- If the instruction asks for clothing, change the clothing while
  keeping the person's face, pose, body proportions and identity
  consistent.
- If the instruction asks for style, apply the style while preserving
  the main subject's identity and composition when possible.

USER INSTRUCTION:
"""

    return (
        base_instruction.strip()
        + "\n"
        + prompt.strip()
    )


# =========================================================
# HUGGING FACE CLIENT
# =========================================================

def get_hf_client():

    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not configured. "
            "Add HF_TOKEN to Render Environment Variables."
        )

    return InferenceClient(
        provider=HF_PROVIDER,
        api_key=HF_TOKEN
    )


# =========================================================
# AI IMAGE EDIT
# =========================================================

def edit_with_ai(
    image_bytes: bytes,
    prompt: str
) -> bytes:

    client = get_hf_client()

    final_prompt = build_edit_prompt(prompt)

    logger.info(
        "Editing image with model=%s provider=%s",
        MODEL,
        HF_PROVIDER
    )

    last_error = None

    # Retry a failed provider request.
    for attempt in range(1, 4):

        try:

            result = client.image_to_image(
                image_bytes,
                prompt=final_prompt,
                model=MODEL
            )

            if result is None:
                raise RuntimeError(
                    "The AI provider returned no image."
                )

            # huggingface_hub normally returns a PIL Image.
            if isinstance(result, Image.Image):

                output = io.BytesIO()

                result.convert("RGB").save(
                    output,
                    format="JPEG",
                    quality=95
                )

                return output.getvalue()

            # Safety fallback if provider returns bytes.
            if isinstance(result, bytes):
                return result

            # Some versions can return file-like objects.
            if hasattr(result, "read"):

                data = result.read()

                if data:
                    return data

            raise RuntimeError(
                f"Unsupported AI response type: "
                f"{type(result).__name__}"
            )

        except Exception as e:

            last_error = e

            logger.exception(
                "AI edit attempt %s failed",
                attempt
            )

            if attempt < 3:
                time.sleep(2 * attempt)

    raise RuntimeError(
        f"AI image editing failed after 3 attempts: "
        f"{last_error}"
    )


# =========================================================
# IMAGE UPLOAD
# =========================================================

def upload_image(image_bytes: bytes) -> str:

    # -----------------------------------------------------
    # Catbox
    # -----------------------------------------------------

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=True
        ) as temp:

            temp.write(image_bytes)
            temp.flush()

            with open(
                temp.name,
                "rb"
            ) as file:

                response = requests.post(
                    "https://catbox.moe/user/api.php",
                    data={
                        "reqtype": "fileupload"
                    },
                    files={
                        "fileToUpload": file
                    },
                    timeout=40
                )

        if (
            response.status_code == 200
            and response.text.strip().startswith("http")
        ):
            return response.text.strip()

    except Exception as e:

        logger.warning(
            "Catbox upload failed: %s",
            e
        )

    # -----------------------------------------------------
    # tmpfiles fallback
    # -----------------------------------------------------

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".jpg",
            delete=True
        ) as temp:

            temp.write(image_bytes)
            temp.flush()

            with open(
                temp.name,
                "rb"
            ) as file:

                response = requests.post(
                    "https://tmpfiles.org/api/v1/upload",
                    files={
                        "file": file
                    },
                    timeout=40
                )

        response.raise_for_status()

        data = response.json()

        url = data["data"]["url"]

        return url.replace(
            "tmpfiles.org/",
            "tmpfiles.org/dl/"
        )

    except Exception as e:

        logger.exception(
            "All image upload methods failed."
        )

        raise RuntimeError(
            f"Image upload failed: {e}"
        )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/")
def home():

    return {
        "status": "online",
        "service": APP_NAME,
        "version": "2.0.0",
        "model": MODEL,
        "provider": HF_PROVIDER,
        "editing": "natural-language"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "hf_configured": bool(HF_TOKEN),
        "model": MODEL
    }


# =========================================================
# MAIN PHOTO EDIT ENDPOINT
# =========================================================

@app.post("/edit-photo/")
def edit_photo(req: EditRequest):

    started = time.time()

    try:

        # -----------------------------------------------
        # 1. Translate prompt
        # -----------------------------------------------

        translated_prompt = translate_prompt(
            req.prompt
        )

        logger.info(
            "Prompt: %s",
            translated_prompt
        )

        # -----------------------------------------------
        # 2. Download image
        # -----------------------------------------------

        original_bytes = download_image(
            req.image_url
        )

        # -----------------------------------------------
        # 3. Normalize image
        # -----------------------------------------------

        image_bytes = normalize_image(
            original_bytes
        )

        # -----------------------------------------------
        # 4. AI EDIT
        # -----------------------------------------------

        edited_bytes = edit_with_ai(
            image_bytes,
            translated_prompt
        )

        # -----------------------------------------------
        # 5. Upload result
        # -----------------------------------------------

        final_url = upload_image(
            edited_bytes
        )

        elapsed = round(
            time.time() - started,
            2
        )

        return {
            **success_response(
                photo_url=final_url,
                original_prompt=req.prompt,
                translated_prompt=translated_prompt
            ),
            "processing_time_seconds": elapsed
        }

    except HTTPException:
        raise

    except Exception as e:

        logger.exception(
            "Photo editing failed"
        )

        return error_response(
            str(e)
        )


# =========================================================
# RUN LOCALLY
# =========================================================

if __name__ == "__main__":

    import uvicorn

    port = int(
        os.getenv("PORT", "8000")
    )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
