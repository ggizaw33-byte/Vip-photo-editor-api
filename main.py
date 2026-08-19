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

def upload_to_tmpfiles(file_path):
    # የተስተካከለውን ፎቶ ወደ ነፃ ሊንክ መቀየሪያ
    with open(file_path, "rb") as f:
        upload_resp = requests.post("https://tmpfiles.org/api/v1/upload", files={"file": f})
        data = upload_resp.json()
        raw_url = data["data"]["url"]
        return raw_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")

@app.post("/edit-photo/")
def edit_photo(req: EditRequest):
    temp_path = None
    try:
        # 1. ትዕዛዙን ወደ እንግሊዝኛ መተርጎም
        translated_prompt = GoogleTranslator(source='auto', target='en').translate(req.prompt).lower()

        # 2. ፎቶውን ከቴሌግራም ዳውንሎድ ማድረግ
        img_resp = requests.get(req.image_url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(img_resp.content)
            temp_path = temp_file.name

        result_path = ""
        
        # 3. AI መምረጥ (VIP Logic)
        # ተጠቃሚው 'background' ማጥፋት ከፈለገ ትክክለኛ Background Remover AI ይጠቀማል
        if "background" in translated_prompt and ("remove" in translated_prompt or "delete" in translated_prompt or "clear" in translated_prompt):
            bg_client = Client("briaai/BRIA-RMBG-1.4")
            result = bg_client.predict(image=handle_file(temp_path), api_name="/predict")
            result_path = result if isinstance(result, str) else result[0]
            
        # ሌላ የፎቶ ኤዲቲንግ ከሆነ አጠቃላይ ኤዲተር AI ይጠቀማል
        else:
            ai_client = Client("timbrooks/instruct-pix2pix")
            result = ai_client.predict(
                translated_prompt,          # ትዕዛዝ
                handle_file(temp_path),     # ፎቶ
                5.5,                        # Text CFG
                1.5,                        # Image CFG
                fn_index=0                  # ስህተት እንዳያመጣ ቀጥታ ኢንዴክስ መጠቀም
            )
            result_path = result[0] if isinstance(result, (list, tuple)) else result

        # 4. የተስተካከለውን ፎቶ ወደ ሊንክ ቀይሮ መመለስ
        direct_url = upload_to_tmpfiles(result_path)

        return {
            "status": "success",
            "photo_url": direct_url
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
