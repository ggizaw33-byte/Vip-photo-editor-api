import os
import tempfile
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from deep_translator import GoogleTranslator
from gradio_client import Client, handle_file

app = FastAPI(title="VIP AI Photo Editor API")

# Background Remover Client
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
            timeout=40
        )
        if resp.status_code == 200 and resp.text.startswith("http"):
            return resp.text.strip()
        else:
            f.seek(0)
            upload_resp = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": f}, timeout=40)
            data = upload_resp.json()
            return data["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")

@app.post("/edit-photo/")
def edit_photo(req: EditRequest):
    temp_path = None
    try:
        # 1. ትዕዛዙን መተርጎም
        translated_prompt = GoogleTranslator(source='auto', target='en').translate(req.prompt).strip()
        lower_prompt = translated_prompt.lower()

        # 2. ፎቶውን ከቴሌግራም ማውረድ
        img_resp = requests.get(req.image_url, timeout=30)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(img_resp.content)
            temp_path = temp_file.name

        result_path = ""

        # 3. ባክግራውንድ ለማጥፋት (Background Removal)
        if "background" in lower_prompt and any(w in lower_prompt for w in ["remove", "delete", "clear", "cut", "no"]):
            result = bg_client.predict(
                handle_file(temp_path),
                api_name="/predict"
            )
            result_path = result if isinstance(result, str) else result[0]

        # 4. ፊትን ጠብቆ ኤዲት ለማድረግ (Stable Face-Preserving Image-to-Image)
        else:
            editor = Client("Zero-GPU-Explorers/FLUX.1-dev-Inpainting")
            result = editor.predict(
                image=handle_file(temp_path),
                prompt=f"{translated_prompt}, high resolution, maintain facial features",
                api_name="/predict"
            )
            result_path = result if isinstance(result, str) else result[0]

        # 5. የተስተካከለውን ፎቶ ሊንክ ማመንጨት
        final_url = upload_image(result_path)

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
