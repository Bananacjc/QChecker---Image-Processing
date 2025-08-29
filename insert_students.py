# insert_students.py
import os, json, math
import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials, firestore

# ---------- Firebase init (env var or local json) ----------
def get_db():
    # Load variables from the .env file into the environment
    load_dotenv(dotenv_path='./.env')

    # Get the FIREBASE_CREDENTIALS variable
    raw = os.environ.get("FIREBASE_CREDENTIALS")
    if not raw:
        raise RuntimeError("Missing FIREBASE_CREDENTIALS in your .env file")

    # Parse JSON and fix private key newlines
    cfg = json.loads(raw)
    cfg["private_key"] = cfg["private_key"].replace("\n", "\n")

    if not firebase_admin._apps:
        cred = credentials.Certificate(cfg)
        firebase_admin.initialize_app(cred)

    return firestore.client()

# ---------- Hard-coded student records ----------
STUDENTS = [
    {
        "seat_num": 1,  # number in Firestore (better for sorting)
        "student_id": "24WMR09274",
        "name": "Tan Jee Cheng",
        "course": "Bachelor of Software Engineering",
        "award": "Graduated with Distinction",
        "image_path": "24WMR09274.png",
        "registered": False,
    },
    {
        "seat_num": 2,
        "student_id": "24WMR09155",
        "name": "Cheong Jau Chun",
        "course": "Bachelor of Computer Science",
        "award": "Book Prize Winner",
        "image_path": "24WMR09155.jpg",
        "registered": False,
    },
    {
        "seat_num": 3,
        "student_id": "24WMR09165",
        "name": "Chong Zhen Yue",
        "course": "Bachelor of Software Engineering",
        "award": "Dean's List",
        "image_path": "24WMR09165.png",
        "registered": False,
    },
    {
        "seat_num": 4,
        "student_id": "24WMR16715",
        "name": "Tan Chun Keat",
        "course": "Bachelor of Software Development",
        "award": "Dean's List",
        "image_path": "24WMR16715.png",
        "registered": False,
    },
]

# ---------- Upsert helper (batch writes) ----------
def upsert_students(students):
    db = get_db()
    coll = db.collection("students")

    # Firestore batch limit is 500 ops; we’ll chunk to be safe
    CHUNK = 400
    total = 0

    for i in range(0, len(students), CHUNK):
        batch = db.batch()
        chunk = students[i : i + CHUNK]

        for s in chunk:
            # sanitize / defaults
            s = dict(s)  # copy
            s["seat_num"] = int(s.get("seat_num", 0))
            s.setdefault("registered", False)
            sid = s.get("student_id")
            if not sid:
                raise ValueError("Each student must include 'student_id'.")

            # use student_id as the document ID (prevents duplicates)
            doc_ref = coll.document(sid)
            batch.set(doc_ref, s, merge=True)  # upsert

        batch.commit()
        total += len(chunk)
        print(f"Committed {total}/{len(students)} records…")

    print("Done. 🎉")

if __name__ == "__main__":
    upsert_students(STUDENTS)
