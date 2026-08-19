import os
import base64
import tempfile
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from deep_translator import GoogleTranslator
from gradio_client import Client, handle_file

app = FastAPI(title="VIP AI Photo Editor API")

# Instruct-Pix2Pix AI Model
ai_client = Client("timbrooks/instruct-pix2pix")

class EditRequest(BaseModel):
    image_url: str
    prompt: str

@app.get("/")
def home():
    return {"status": "VIP Photo Editor API is running successfully!"}

@app.post("/edit-photo/")
def edit_photo(req: EditRequest):
    temp_path = None
    try:
        # 1. ፕሮምፕቱን ወደ እንግሊዝኛ መተርጎም
        translated_prompt = GoogleTranslator(source='auto', target='en').translate(req.prompt)

        # 2. ፎቶውን ከቴሌግራም ማውረድ
        img_resp = requests.get(req.image_url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(img_resp.content)
            temp_path = temp_file.name

        # 3. AI ሞዴሉ እንዲያስተካክለው መላክ
        result_path = ai_client.predict(
            image=handle_file(temp_path),
            prompt=translated_prompt,
            num_inference_steps=20,
            image_guidance_scale=1.5,
            guidance_scale=7.5,
            api_name="/predict"
        )

        # 4. የተስተካከለውን ፎቶ ወደ Base64 መቀየር
        with open(result_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf-8")

        return {
            "status": "success",
            "image_base64": f"data:image/jpeg;base64,{encoded_image}"
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
