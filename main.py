import os
import tempfile
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from deep_translator import GoogleTranslator
from gradio_client import Client, handle_file

app = FastAPI(title="VIP AI Photo Editor API")

class EditRequest(BaseModel):
    image_url: str
    prompt: str

@app.get("/")
def home():
    return {"status": "VIP Photo Editor API is running successfully!"}

def upload_image(file_path):
    # ቴሌግራም ያለምንም ችግር የሚያነበው ፈጣን Image Host (Catbox)
    with open(file_path, "rb") as f:
        resp = requests.post(
            "https://catbox.moe/user/api.php",
            data={"reqtype": "fileupload"},
            files={"fileToUpload": f}
        )
        if resp.status_code == 200 and resp.text.startswith("http"):
            return resp.text.strip()
        else:
            # አማራጭ Host (tmpfiles)
            f.seek(0)
            upload_resp = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": f})
            data = upload_resp.json()
            return data["data"]["url"].replace("tmpfiles.org/", "tmpfiles.org/dl/")

@app.post("/edit-photo/")
def edit_photo(req: EditRequest):
    temp_path = None
    try:
        # 1. ማንኛውንም ቋንቋ ወደ እንግሊዝኛ መተርጎም
        translated_prompt = GoogleTranslator(source='auto', target='en').translate(req.prompt).strip()

        # 2. ፎቶውን ከቴሌግራም ማውረድ
        img_resp = requests.get(req.image_url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(img_resp.content)
            temp_path = temp_file.name

        lower_prompt = translated_prompt.lower()
        result_path = ""

        # 3. ባክግራውንድ ለማጥፋት (Background Removal)
        if "background" in lower_prompt and ("remove" in lower_prompt or "delete" in lower_prompt or "clear" in lower_prompt or "cut" in lower_prompt):
            bg_client = Client("briaai/BRIA-RMBG-1.4")
            result = bg_client.predict(handle_file(temp_path))
            result_path = result if isinstance(result, str) else result[0]

        # 4. አጠቃላይ AI ፎቶ ኤዲቲንግ (Image Editing)
        else:
            ai_client = Client("timbrooks/instruct-pix2pix")
            result = ai_client.predict(
                translated_prompt,
                handle_file(temp_path),
                7.5,
                1.5
            )
            result_path = result[0] if isinstance(result, (list, tuple)) else result

        # 5. የተስተካከለውን ፎቶ ቀጥታ ሊንክ ማግኘት
        direct_url = upload_image(result_path)

        return {
            "status": "success",
            "photo_url": direct_url
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
