import os
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
        # Catbox ላይ መጫን
        resp = requests.post(
            "https://catbox.moe/user/api.php",
            data={"reqtype": "fileupload"},
            files={"fileToUpload": f},
            timeout=30
        )
        if resp.status_code == 200 and resp.text.startswith("http"):
            return resp.text.strip()
        else:
            # አማራጭ Host (tmpfiles)
            f.seek(0)
            upload_resp = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": f}, timeout=30)
            data = upload_resp.json()
            return data["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")

@app.post("/edit-photo/")
def edit_photo(req: EditRequest):
    temp_path = None
    try:
        # 1. ማንኛውንም ቋንቋ ወደ እንግሊዝኛ መተርጎም
        translated_prompt = GoogleTranslator(source='auto', target='en').translate(req.prompt).strip()
        lower_prompt = translated_prompt.lower()

        # 2. ፎቶውን ከቴሌግራም ማውረድ
        img_resp = requests.get(req.image_url, timeout=30)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(img_resp.content)
            temp_path = temp_file.name

        result_path = ""

        # 3. ባክግራውንድ ለማጥፋት ከሆነ
        if "background" in lower_prompt and any(w in lower_prompt for w in ["remove", "delete", "clear", "cut", "no"]):
            result = bg_client.predict(
                handle_file(temp_path),
                api_name="/predict"
            )
            result_path = result if isinstance(result, str) else result[0]
            final_url = upload_image(result_path)

        # 4. ለማንኛውም ሌላ የፎቶ ትዕዛዝ (All AI Image-to-Image Prompts)
        else:
            # ፎቶውን በነፃ እና ፈጣን በሆነው Pix2Pix / IP-Adapter ሞዴል ማስተካከል
            edit_client = Client("ysharma/InstructPix2Pix_Fast")
            result = edit_client.predict(
                handle_file(temp_path),    # ፎቶ
                translated_prompt,         # ትዕዛዝ
                1.5,                       # Image guidance (ፊቱን ለመጠበቅ)
                7.5,                       # Text guidance
                fn_index=0
            )
            result_path = result if isinstance(result, str) else result[0]
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
