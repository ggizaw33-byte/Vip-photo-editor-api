import io
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from deep_translator import GoogleTranslator
from gradio_client import Client, handle_file

app = FastAPI(title="VIP AI Photo Editor API")

# Image-to-Image AI Client
ai_client = Client("timbrooks/instruct-pix2pix")

@app.get("/")
def home():
    return {"status": "VIP Photo Editor API is running successfully!"}

@app.post("/edit-photo/")
async def edit_photo(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    temp_path = None
    try:
        # 1. ማንኛውንም ቋንቋ ወደ እንግሊዝኛ መተርጎም
        translated_prompt = GoogleTranslator(source='auto', target='en').translate(prompt)

        # 2. ፎቶውን ለጊዜው ማስቀመጥ
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        # 3. AI ሞዴሉ ፊትን ጠብቆ ኤዲት እንዲያደርግ መላክ
        result = ai_client.predict(
            image=handle_file(temp_path),
            prompt=translated_prompt,
            num_inference_steps=20,
            image_guidance_scale=1.5,
            guidance_scale=7.5,
            api_name="/predict"
        )

        # 4. የተስተካከለውን ፎቶ ማንበብና መመለስ
        with open(result, "rb") as f:
            image_bytes = f.read()

        return Response(content=image_bytes, media_type="image/jpeg")

    except Exception as e:
        return {"error": str(e)}

    finally:
        # ጊዜያዊ ፋይሉን ማጽዳት
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
