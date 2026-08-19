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
    # ፎቶውን ወደ ሊንክ ለመቀየር (በ Catbox እና Tmpfiles)
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
        # 1. የገባውን ፕሮምፕት ወደ እንግሊዝኛ መተርጎም
        translated_prompt = GoogleTranslator(source='auto', target='en').translate(req.prompt).strip()
        lower_prompt = translated_prompt.lower()

        # 2. ፎቶውን ዳውንሎድ ማድረግ
        img_resp = requests.get(req.image_url, timeout=30)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(img_resp.content)
            temp_path = temp_file.name

        result_path = ""

        # 3. ባክግራውንድ ለማጥፋት ከሆነ (ይሄ በትክክል ይሰራል)
        if "background" in lower_prompt and any(w in lower_prompt for w in ["remove", "delete", "clear", "cut", "no"]):
            # Lazy Load - Renderን እንዳያጨናንቀው
            bg_client = Client("briaai/BRIA-RMBG-1.4")
            result = bg_client.predict(handle_file(temp_path), api_name="/predict")
            result_path = result if isinstance(result, str) else result[0]

        # 4. ለማንኛውም ሌላ የፎቶ ትዕዛዝ (All Prompts - ልብስ፣ ከለር፣ ስታይል ወዘተ)
        else:
            # Lazy Load - ዋናውን InstructPix2Pix መጠቀም
            edit_client = Client("timbrooks/instruct-pix2pix")
            
            # ተጠቃሚው "change face" ካላለ፣ ፊቱን እንዳይቀይር ለ AI ጥብቅ ትዕዛዝ መስጠት
            if "change face" not in lower_prompt and "replace face" not in lower_prompt:
                final_prompt = f"{translated_prompt}, strictly preserve exact facial identity, keep original face, do not modify facial features"
                img_guidance = 1.8  # ፊቱን እና ዋናውን ቅርጽ አጥብቆ እንዲይዝ ያደርጋል
            else:
                # ፊቱን እንዲቀይር ከተፈቀደለት
                final_prompt = translated_prompt
                img_guidance = 1.2  # ብዙ ለውጥ እንዲያደርግ ይፈቀድለታል

            # AI ኤዲት እንዲያደርገው መላክ
            result = edit_client.predict(
                final_prompt,
                handle_file(temp_path),
                7.5,              # Text Guidance
                img_guidance,     # Image Guidance (ፊቱን ለመጠበቅ)
                fn_index=0        # Endpoint ስህተት እንዳያመጣ
            )
            result_path = result[0] if isinstance(result, (list, tuple)) else result

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
