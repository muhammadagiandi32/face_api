# face_verification_service.py
"""
FastAPI backend sederhana untuk verifikasi wajah
================================================

Business Rule (Final)
---------------------
- Tabel `users` menyimpan NIP dan encoding wajah.
- Endpoint `/verify/` menerima `nip` dan `file`.
  - Jika NIP belum terdaftar, maka user dan encoding disimpan.
  - Jika NIP sudah ada, maka gambar dicocokkan dengan encoding yang tersimpan.
  - Response: "matched" atau "not matched" (dalam JSON).
"""


import io
import pickle
from typing import List

import numpy as np
import face_recognition
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from PIL import Image, ExifTags
from sqlalchemy import create_engine, Column, String, BLOB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from fastapi.responses import JSONResponse

app = FastAPI(title="Face Verification Service With DB")
THRESHOLD: float = 0.38  # face distance threshold

# DATABASE SETUP
DATABASE_URL = "mysql+mysqlconnector://admin_user:password_kamu@localhost:3306/absensi_db"
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


class User(Base):
    __tablename__ = "users"
    nip = Column(String(50), primary_key=True, nullable=False)
    encoding = Column(BLOB, nullable=False)  # pickled list[np.ndarray]


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_image(bytes_data: bytes) -> Image.Image:
    """Buka gambar, putar berdasarkan EXIF, dan resize jika perlu."""
    image = Image.open(io.BytesIO(bytes_data))
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = dict(image._getexif().items())
        if exif.get(orientation) == 3:
            image = image.rotate(180, expand=True)
        elif exif.get(orientation) == 6:
            image = image.rotate(270, expand=True)
        elif exif.get(orientation) == 8:
            image = image.rotate(90, expand=True)
    except Exception:
        pass
    if max(image.size) > 1024:
        image.thumbnail((1024, 1024))
    return image


def get_encodings(image: Image.Image) -> List[np.ndarray]:
    rgb = np.array(image.convert("RGB"))
    return face_recognition.face_encodings(rgb)


@app.post("/absen/")
async def verify_face(
    nip: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    # 1. Baca gambar & ambil encoding wajah
    data = await file.read()
    image = get_image(data)
    encs = get_encodings(image)
    if not encs:
        raise HTTPException(400, "Tidak ada wajah terdeteksi.")
    target = encs[0]

    # 2. Cek apakah NIP sudah terdaftar
    user = db.query(User).filter_by(nip=nip).first()

    if user is None:
        # → Simpan user baru jika belum ada
        db.add(User(nip=nip, encoding=pickle.dumps([target])))
        db.commit()
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "message": "User baru didaftarkan dan absensi direkam.",
                "data": {
                    "nip": nip,
                    "matched": True,
                    "result": "registered"
                }
            }
        )

    # 3. Bandingkan wajah jika NIP sudah terdaftar
    known_list = pickle.loads(user.encoding)
    known_enc = known_list[0] if isinstance(known_list, list) else known_list
    dist = float(face_recognition.face_distance([known_enc], target)[0])
    matched = bool(dist < THRESHOLD)

    return JSONResponse(
            status_code=401,
            content={
                "status": "failed",
                "message": "Wajah tidak cocok dengan data yang terdaftar.",
                "data": {
                    "nip": nip,
                    "matched": matched,
                    "distance": dist,
                    "threshold": THRESHOLD,
                    "result": "not matched",
                },
            },
        )


@app.get("/")
def root():
    return {"msg": "Face Verification Service with Database running"}
