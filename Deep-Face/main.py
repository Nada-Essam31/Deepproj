from build_face_database import load_face_database
from find_face import find_faces
from display_results import display_results
from get_name import extract_names_from_results 
from datetime import datetime , date
import pandas as pd
import os
import cv2
db_file = "face_db.pkl"  
face_db = load_face_database(db_file)

img_path = "C2202210.jpeg"  # Your test image

results = find_faces(
    img_path=img_path,
    face_db=face_db,
    threshold=0.6  # Adjust threshold as needed
)

# Display the results with bounding boxes and labels
display_results(
    img_path=img_path, 
    results=results, 
    confidence_threshold=0.4
)


Id = extract_names_from_results(results)
# ✅ نضمن إن Id قائمة حتى لو فيها اسم واحد
if isinstance(Id, str):
    Id = [Id]

print("Matched Names:", Id)

print("ID contents:", Id, type(Id))

# إعداد ملف Excel
today = date.today().strftime("%Y-%m-%d")
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
excel_path = fr"C:\Users\nadae\Documents\attendance_{today}.xlsx"

# تحميل أو إنشاء الملف
if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
else:
    df = pd.DataFrame(columns=["Student ID", "Time"])

# تسجيل الحضور
for student_id in Id:
    # تأكد إنه string سليم
    student_id = str(student_id).strip()

    # تجنب التكرار في نفس اليوم
    if not ((df["Student ID"] == student_id) & (df["Time"].str.startswith(today))).any():
        df.loc[len(df)] = [student_id, current_time]
        print(f"✅ تم تسجيل {student_id} في {current_time}")

# حفظ التحديث
df.to_excel(excel_path, index=False)
print("✅ Attendance saved to:", excel_path)


# id -get name from database - get time - save excel sheet