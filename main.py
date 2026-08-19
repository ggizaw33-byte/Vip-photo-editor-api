import os
import tempfile
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from deep_translator import GoogleTranslator
from gradio_client import Client, handle_file

app = FastAPI(title="VIP AI Photo Editor API")

# ሞዴሎችን አስቀድሞ ማዘጋጀት
bg_client = Client("briaai/BRIA-RMBG-1.4")
edit_client = Client("timbrooks/instruct-pix2pix")

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
            # ካልሰራ በ tmpfiles አማራጭ
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

        # 3. ባክግራውንድ ብቻ ለማጥፋት ከሆነ
        if "background" in lower_prompt and any(w in lower_prompt for w in ["remove", "delete", "clear", "cut", "no"]):
            result = bg_client.predict(
                handle_file(temp_path),
                fn_index=0
            )
            result_path = result if isinstance(result, str) else result[0]

        # 4. ለማንኛውም ሌላ የፎቶ ኤዲቲንግ ትዕዛዝ (All Prompts)
        else:
            result = edit_client.predict(
                translated_prompt,          # የ AI ትዕዛዝ
                handle_file(temp_path),     # የተጠቃሚው ፎቶ
                7.5,                        # Text guidance scale
                1.5,                        # Image guidance (ፊቱን እና ቅርጹን ለመጠበቅ)
                fn_index=0                  # ስህተቱን የሚያጠፋው ዋና ቁልፍ
            )
            result_path = result[0] if isinstance(result, (list, tuple)) else result

        # 5. የተስተካከለውን ፎቶ ሊንክ ማመንጨት
        direct_url = upload_image(result_path)

        return {
            "status": "success",
            "photo_url": direct_url,
            "translated_prompt": translated_prompt
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
