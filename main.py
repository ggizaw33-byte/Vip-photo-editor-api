import os
import urllib.parse
import tempfile
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from deep_translator import GoogleTranslator
from gradio_client import Client, handle_file

app = FastAPI(title="VIP AI Photo Editor API")

# Background Remover
bg_client = Client("briaai/BRIA-RMBG-1.4")

class EditRequest(BaseModel):
    image_url: str
    prompt: str

@app.get("/")
def home():
    return {"status": "VIP Photo Editor API is running successfully!"}

def upload_image(file_path):
    with open(file_path, "rb") as f:
        resp = requests.post(
            "https://catbox.moe/user/api.php",
            data={"reqtype": "fileupload"},
            files={"fileToUpload": f},
            timeout=30
        )
        if resp.status_code == 200 and resp.text.startswith("http"):
            return resp.text.strip()
        else:
            f.seek(0)
            upload_resp = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": f}, timeout=30)
            data = upload_resp.json()
            return data["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")

@app.post("/edit-photo/")
def edit_photo(req: EditRequest):
    temp_path = None
    try:
        # 1. ፕሮምፕቱን ወደ እንግሊዝኛ መተርጎም
        translated_prompt = GoogleTranslator(source='auto', target='en').translate(req.prompt).strip()
        lower_prompt = translated_prompt.lower()

        # 2. ባክግራውንድ ብቻ ለማጥፋት ከሆነ (Background Removal)
        if "background" in lower_prompt and any(w in lower_prompt for w in ["remove", "delete", "clear", "cut", "no"]):
            img_resp = requests.get(req.image_url, timeout=30)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                temp_file.write(img_resp.content)
                temp_path = temp_file.name

            result = bg_client.predict(handle_file(temp_path), api_name="/predict")
            result_path = result if isinstance(result, str) else result[0]
            final_url = upload_image(result_path)

        # 3. ለማንኛውም ሌላ የፎቶ ትዕዛዝ (Unlimited VIP Image-to-Image AI)
        else:
            # ፊትን እና የመጀመሪያውን ፎቶ ዝርዝር ጠብቆ ኤዲት የሚያደርግ የተረጋጋ የ AI ትዕዛዝ
            enhanced_prompt = f"{translated_prompt}, preserve exact facial features, high quality, 8k photo"
            encoded_prompt = urllib.parse.quote(enhanced_prompt)
            encoded_image_url = urllib.parse.quote(req.image_url)

            # Unlimited Pollinations AI Endpoint
            final_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?image={encoded_image_url}&model=flux&nologo=true"

        return {
            "status": "success",
            "photo_url": final_url,
            "translated_prompt": translated_prompt
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
