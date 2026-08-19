import os
import tempfile
import requests
from PIL import Image
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

        # 2. የተጠቃሚውን ፎቶ ማውረድ
        img_resp = requests.get(req.image_url, timeout=30)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file.write(img_resp.content)
            temp_path = temp_file.name

        result_path = ""
        bg_client = Client("briaai/BRIA-RMBG-1.4")

        # ሁኔታ 1፡ ባክግራውንድ ብቻ ማጥፋት (Remove background)
        if "background" in lower_prompt and any(w in lower_prompt for w in ["remove", "delete", "clear", "cut", "no"]):
            result = bg_client.predict(handle_file(temp_path), api_name="/predict")
            result_path = result if isinstance(result, str) else result[0]

        # ሁኔታ 2፡ ባክግራውንድ መቀየር (Change background to sky, garden, beach, studio...)
        elif "background" in lower_prompt or "bg" in lower_prompt:
            # ሀ. ልጁን ከፎቶው ቆርጦ ማውጣት
            cutout_res = bg_client.predict(handle_file(temp_path), api_name="/predict")
            cutout_path = cutout_res if isinstance(cutout_res, str) else cutout_res[0]
            
            # ለ. አዲሱን ባክግራውንድ በ AI ማመንጨት
            bg_query = translated_prompt.replace("change background to", "").replace("background", "").strip()
            bg_url = f"https://image.pollinations.ai/prompt/scenic%20photorealistic%20background%20of%20{bg_query}%20no%20people?width=768&height=1024&nologo=true"
            bg_data = requests.get(bg_url, timeout=30).content
            
            # ሐ. ሁለቱን ፎቶዎች አንድ ላይ ማዋሃድ (Composite)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as out_file:
                bg_img = Image.open(requests.get(bg_url, stream=True).raw).convert("RGBA")
                person_img = Image.open(cutout_path).convert("RGBA")
                
                bg_img = bg_img.resize(person_img.size)
                bg_img.paste(person_img, (0, 0), person_img)
                
                final_output = bg_img.convert("RGB")
                final_output.save(out_file.name, format="JPEG", quality=95)
                result_path = out_file.name

        # ሁኔታ 3፡ ሌሎች የፎቶ ማስተካከያዎች (All Other Prompts - ፊትን ጠብቆ)
        else:
            # ፊትን ሳያበላሽ ልብስ/ስታይል የሚቀይር ፈጣን ሞዴል
            face_client = Client("lambdalabs/instruct-pix2pix")
            result = face_client.predict(
                input_image=handle_file(temp_path),
                instruction=f"{translated_prompt}, preserve facial identity",
                steps=20,
                randomize_seed=True,
                seed=42,
                text_cfg=7.5,
                image_cfg=1.8,
                api_name="/predict"
            )
            result_path = result if isinstance(result, str) else result[0]

        # ፎቶውን ወደ ሊንክ መቀየር
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
