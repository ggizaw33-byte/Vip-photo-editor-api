import os
import tempfile
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from deep_translator import GoogleTranslator
from gradio_client import Client, handle_file

app = FastAPI(title="VIP AI Photo Editor API")

# ይበልጥ ፈጣንና ቋሚ የሆነ Face-Preserving Image Editor Client
ai_client = Client("radames/Enhance-This-Edit-That")

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
        # 1. ማንኛውንም ቋንቋ ወደ እንግሊዝኛ መተርጎም
        translated_prompt = GoogleTranslator(source='auto', target='en').translate(req.prompt)

        # 2. ፎቶውን ከቴሌግራም ማውረድ
        img_resp = requests.get(req.image_url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(img_resp.content)
            temp_path = temp_file.name

        # 3. AI ሞዴሉ እንዲያስተካክለው መላክ (ያለ api_name በቀጥታ)
        result = ai_client.predict(
            input_image=handle_file(temp_path),
            instruction=translated_prompt
        )

        result_path = result if isinstance(result, str) else result[0]

        # 4. የተስተካከለውን ፎቶ ለቴሌግራም ክፍት ወደሆነ ነፃ Image Host መስቀል
        with open(result_path, "rb") as f:
            upload_resp = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": f})
            data = upload_resp.json()
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
