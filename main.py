import io
import uuid
import base64
import pickle
import numpy as np
from PIL import Image, ExifTags
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy import create_engine, Column, String, Text, BLOB, Integer, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import face_recognition
from datetime import datetime, timedelta
from jose import JWTError, jwt
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import bcrypt
from fastapi import Body
from sqlalchemy import and_

DATABASE_URL = "mysql+mysqlconnector://admin_user:password_kamu@115.124.68.196:3306/kehadiran"
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

SECRET_KEY = "YOUR_SECRET_KEY_GANTI_SENDIRI"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 120

class User(Base):
    __tablename__ = "users"
    id = Column(String(6), primary_key=True, nullable=False)
    nama = Column(String(100), nullable=False)
    email = Column(String(100), nullable=False, unique=True)
    password = Column(String(255), nullable=False)
    foto_registrasi = Column(Text)
    encoding = Column(BLOB)

class Absen(Base):
    __tablename__ = "absen"
    no = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(String(6), nullable=False)
    datetime = Column(String(25), nullable=False)
    date = Column(Date, nullable=False)
    time = Column(String(10), nullable=False)
    aut = Column(String(50), nullable=False)
    foto = Column(Text, nullable=False)
    latitude = Column(String(50))
    longitude = Column(String(50))
    lokasi = Column(Text)
    encoding = Column(BLOB)

Base.metadata.create_all(bind=engine)

app = FastAPI()
THRESHOLD = 0.38

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_image(bytes_data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(bytes_data))
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
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

def get_encodings(image: Image.Image) -> list:
    rgb = np.array(image.convert("RGB"))
    return face_recognition.face_encodings(rgb)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_password(plain, hashed):
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


bearer_scheme = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("email")
        if email is None:
            raise HTTPException(401, "Token tidak valid!")
        user = db.query(User).filter_by(email=email).first()
        if not user:
            raise HTTPException(401, "User tidak ditemukan!")
        return user
    except JWTError:
        raise HTTPException(401, "Token tidak valid!")

@app.post("/login/")
async def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=email).first()
    if not user or not verify_password(password, user.password):
        raise HTTPException(401, "Email atau password salah!")
    token = create_access_token({"email": user.email, "name":user.nama})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "nama": user.nama,
            "email": user.email
        }
    }
@app.post("/register-scan/")
async def register_scan(
    file: UploadFile = File(...),
    nama: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        print(f"[DEBUG] Mulai register untuk: {nama} | {email}")
        # 1. Cek email
        if db.query(User).filter_by(email=email).first():
            print(f"[DEBUG] Email sudah terdaftar: {email}")
            raise HTTPException(409, "Email sudah terdaftar!")
        # 2. Cek nama
        if db.query(User).filter_by(nama=nama).first():
            print(f"[DEBUG] Nama sudah terdaftar: {nama}")
            raise HTTPException(409, "Nama sudah terdaftar!")
        
        # 3. Baca dan proses file
        data = await file.read()
        print(f"[DEBUG] File foto diterima, ukuran: {len(data)} bytes")
        image = get_image(data)
        encs = get_encodings(image)
        if not encs:
            print(f"[DEBUG] Tidak ada wajah terdeteksi dalam gambar")
            raise HTTPException(400, "Tidak ada wajah terdeteksi.")
        face_encoding = encs[0]
        print(f"[DEBUG] Encoding wajah berhasil didapatkan")

        # 4. Cek kemiripan dengan user lain
        all_users = db.query(User).all()
        for user in all_users:
            if user.encoding:
                user_enc = pickle.loads(user.encoding)
                if isinstance(user_enc, list):
                    user_enc = user_enc[0]
                distance = face_recognition.face_distance([np.array(user_enc)], face_encoding)[0]
                print(f"[DEBUG] Cek kemiripan dengan {user.nama}: distance={distance}")
                if distance < THRESHOLD:
                    print(f"[DEBUG] Wajah mirip dengan user lain: {user.nama} | Jarak: {distance}")
                    raise HTTPException(409, f"Wajah sudah terdaftar atas nama {user.nama}!")

        # 5. Hash password
        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        print("HASHED BYTES:", hashed_pw)
        hashed_pw_str = hashed_pw.decode()
        print("HASHED DECODED:", hashed_pw_str)

        # 6. Simpan ke database
        foto_base64 = base64.b64encode(data).decode('utf-8')
        user_id = str(uuid.uuid4())[:6]
        print(f"[DEBUG] User ID: {user_id}")

                
        user = User(
            id=user_id,
            nama=nama,
            email=email,
            password=hashed_pw_str, 
            foto_registrasi=foto_base64,
            encoding=pickle.dumps([face_encoding])
        )
        db.add(user)
        db.commit()
        print(f"[DEBUG] User berhasil didaftarkan dan disimpan ke database.")

        return {
            "status": "completed",
            "progress": 100,
            "instruction": "Registrasi selesai!",
            "user_id": user_id,
        }
    except HTTPException as he:
        print(f"[DEBUG][HTTPException] {he.detail}")
        raise he
    except Exception as e:
        db.rollback()
        print("[DEBUG][DB ERROR]:", str(e))
        raise HTTPException(500, f"DB Error: {e}")

@app.post("/refresh/")
async def refresh_token(email: str = Body(...), db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=email).first()
    if not user:
        raise HTTPException(404, "User tidak ditemukan!")
    # Buat token baru, payload bebas (email, nama, dsb)
    new_token = create_access_token({"email": user.email, "name": user.nama})
    return {
        "access_token": new_token,
        "token_type": "bearer"
    }

@app.post("/absen/")
async def absen(
    file: UploadFile = File(...),
    aut: str = Form(...),
    latitude: str = Form(None),
    longitude: str = Form(None),
    lokasi: str = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        now = datetime.now()
        today = now.date()
        jenis = aut.strip().lower()

        # CEK: sudah absen (masuk/keluar) di hari ini?
        existing = db.query(Absen).filter(
            and_(
                Absen.id == user.id,
                Absen.date == today,
                Absen.aut.ilike(jenis)  # case-insensitive
            )
        ).first()

        if existing:
            raise HTTPException(409, f"Anda Sudah absen {aut} hari ini!")

        # Proses absensi seperti biasa...
        data = await file.read()
        image = get_image(data)
        encs = get_encodings(image)
        if not encs:
            raise HTTPException(400, "Tidak ada wajah terdeteksi.")
        target = encs[0]
        known = pickle.loads(user.encoding)
        if isinstance(known, list):
            known_encs = [np.array(e) for e in known]
        else:
            known_encs = [np.array(known)]
        distances = face_recognition.face_distance(known_encs, target)
        min_distance = float(np.min(distances))
        verified = bool(min_distance < THRESHOLD)
        if not verified:
            raise HTTPException(403, "Wajah tidak cocok.")

        absen_entry = Absen(
            id=user.id,
            datetime=now.strftime("%Y-%m-%d %H:%M:%S"),
            date=today,
            time=now.strftime("%H:%M:%S"),
            aut=aut,
            foto=base64.b64encode(data).decode('utf-8'),
            latitude=latitude,
            longitude=longitude,
            lokasi=lokasi,
            encoding=pickle.dumps([target])
        )
        db.add(absen_entry)
        db.commit()
        return {
            "status": "success",
            "message": f"Absen {aut} berhasil!",
            "absen_no": absen_entry.no,
            "nama": user.nama,
            "waktu": now.strftime("%Y-%m-%d %H:%M:%S"),
        }
    except HTTPException as he:
        print("ABSEN ERROR:", str(he.detail))
        raise he
    except Exception as e:
        db.rollback()
        print("ABSEN ERROR:", str(e))
        raise HTTPException(500, f"Absen error: {e}")
    
@app.post("/verify/")
async def verify_face(
    file: UploadFile = File(...),
    nama: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        user = db.query(User).filter_by(nama=nama).first()
        if not user:
            raise HTTPException(404, f"Nama '{nama}' tidak ditemukan di database.")
        data = await file.read()
        image = get_image(data)
        encs = get_encodings(image)
        if not encs:
            image.save("debug_no_face_verify.jpg")
            raise HTTPException(400, "Tidak ada wajah terdeteksi.")
        target = encs[0]
        known = pickle.loads(user.encoding)
        distances = face_recognition.face_distance([np.array(e) for e in known], target)
        min_distance = float(np.min(distances))
        verified = bool(min_distance < THRESHOLD)
        return {
            "verified": verified,
            "nama": nama,
            "distance": min_distance,
            "threshold": THRESHOLD,
        }
    finally:
        db.close()
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from datetime import datetime

@app.get("/absen/history")
def get_absen_history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    import time
    t0 = time.time()
    try:
        rows = (
            db.query(Absen)
            .filter(Absen.id == user.id)
            .order_by(Absen.date.desc(), Absen.time.desc())
            .all()
        )

        # Group by tanggal
        history = {}
        for ab in rows:
            date_str = ab.date.strftime("%A, %d %B %Y")
            jam = ab.time
            jenis = ab.aut.lower()
            lokasi = ab.lokasi

            if date_str not in history:
                history[date_str] = {"masuk": None, "keluar": None, "lokasi": lokasi}

            if "masuk" in jenis and not history[date_str]["masuk"]:
                history[date_str]["masuk"] = jam
                history[date_str]["lokasi"] = lokasi
            if "keluar" in jenis:
                history[date_str]["keluar"] = jam

        result = []
        for date_str, val in history.items():
            result.append({
                "tanggal": date_str,
                "masuk": val["masuk"],
                "keluar": val["keluar"],
                "lokasi": val["lokasi"],
            })

        result = sorted(result, key=lambda x: datetime.strptime(x["tanggal"], "%A, %d %B %Y"), reverse=True)

        t1 = time.time()
        print(f"[DEBUG] History endpoint selesai dalam {t1-t0:.2f} detik")
        return {"history": result}

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("[ERROR] /absen/history:", str(e))
        print(tb)
        # Kamu bisa return pesan error ke frontend juga
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

