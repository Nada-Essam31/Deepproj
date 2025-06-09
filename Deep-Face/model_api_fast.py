from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.staticfiles import StaticFiles
from build_face_database import load_face_database
from find_face import find_faces
from get_name import extract_names_from_results
from datetime import datetime, date
import pandas as pd
import cv2
import numpy as np
import os
import requests

app = FastAPI()

# إعداد حفظ ملفات الإكسل
EXCEL_DIR = "saved_excels"
os.makedirs(EXCEL_DIR, exist_ok=True)
app.mount("/files", StaticFiles(directory=EXCEL_DIR), name="files")

# تحميل قاعدة الوجوه
face_db = load_face_database("face_db.pkl")

@app.post("/api/attendance/recognize")
async def recognize_face(image: UploadFile = File(...)):
    try:
        # حفظ الصورة المؤقتة
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="❌ لم يتم التعرف على الصورة")

        temp_path = "temp_image.png"
        cv2.imwrite(temp_path, img)

        results = find_faces(temp_path, face_db, threshold=0.6)
        Ids = extract_names_from_results(results)
        if isinstance(Ids, str):
            Ids = [Ids]

        today = date.today().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        excel_filename = f"attendance_{today}.xlsx"
        excel_path = os.path.join(EXCEL_DIR, excel_filename)

        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
        else:
            df = pd.DataFrame(columns=["Student ID", "Time"])

        recognized_data = []

        for student_id in Ids:
            student_id = str(student_id).strip()

            if not ((df["Student ID"] == student_id) & (df["Time"].str.startswith(today))).any():
                df.loc[len(df)] = [student_id, current_time]

            # إرسال للباك إند
            backend_url = "http://127.0.0.1:8001/api/attendance/submit"
            payload = {
                "student_id": student_id,
                "attendance_time": current_time
            }

            try:
                response = requests.post(backend_url, json=payload)
                print("📡 Backend Response:", response.status_code, response.text)
            except Exception as e:
                print("❌ Failed to send to backend:", str(e))

            recognized_data.append({
                "student_id": student_id,
                "attendance_time": current_time
            })

        df.to_excel(excel_path, index=False)
        download_url = f"http://127.0.0.1:8000/files/{excel_filename}"

        return {
            "recognized_ids": recognized_data,
            "excel_download_url": download_url,
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
