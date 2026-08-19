import os
import tempfile
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from deep_translator import GoogleTranslator
from gradio_client import Client, handle_file

app = FastAPI(title="VIP AI Photo Editor API")

# AI Client
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
        # 1. መተርጎም
        translated_prompt = GoogleTranslator(source='auto', target='en').translate(req.prompt)

        # 2. ፎቶውን ማውረድ
        img_resp = requests.get(req.image_url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(img_resp.content)
            temp_path = temp_file.name

        # 3. AI ኤዲት እንዲያደርግ መላክ
        result_path = ai_client.predict(
            image=handle_file(temp_path),
            prompt=translated_prompt,
            num_inference_steps=20,
            image_guidance_scale=1.5,
            guidance_scale=7.5,
            api_name="/predict"
        )

        # 4. የተሰራውን ፎቶ ለቴሌግራም ክፍት ወደሆነ ነፃ Image Hosting መስቀል (ቀጥታ ሊንክ ለመስጠት)
        with open(result_path, "rb") as f:
            upload_resp = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": f})
            data = upload_resp.json()
            # tmpfiles url ወደ ቀጥታ ማውረጃ ሊንክ መቀየር
            raw_url = data["data"]["url"]
            direct_url = raw_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")

        return {
            "status": "success",
            "photo_url": direct_url
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
