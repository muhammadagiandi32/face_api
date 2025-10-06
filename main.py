from dotenv import load_dotenv
load_dotenv()


import os
import io
import uuid
import base64
import pickle
import logging
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image, ExifTags
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Body
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy import create_engine, Column, String, Text, BLOB, Integer, Date, and_, text
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.dialects.mysql import LONGTEXT
from jose import JWTError, jwt
import bcrypt
import uvicorn

# =========================
# LOGGING & KONFIG ENV
# =========================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PORT = int(os.getenv("PORT", 8000))
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+mysqlconnector://wadmin:VWVBP04-HJFq@116.193.191.198:3306/kehadiran"
)
SECRET_KEY = os.getenv("SECRET_KEY", "YOUR_SECRET_KEY_GANTI_SENDIRI")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 120))
THRESHOLD = float(os.getenv("FACE_THRESHOLD", 0.38))


# =========================
# DATABASE SETUP
# =========================
Base = declarative_base()
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
    future=True,
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


# =========================
# MODEL
# =========================
class User(Base):
    __tablename__ = "users"
    id = Column(String(6), primary_key=True, nullable=False)
    nama = Column(String(100), nullable=False)
    email = Column(String(100), nullable=False, unique=True)
    password = Column(String(255), nullable=False)
    foto_registrasi = Column(LONGTEXT)
    encoding = Column(BLOB)


class Absen(Base):
    __tablename__ = "absen"
    no = Column(Integer, primary_key=True, autoincrement=True)
    id = Column(String(6), nullable=False)
    datetime = Column(String(25), nullable=False)
    date = Column(Date, nullable=False)
    time = Column(String(10), nullable=False)
    aut = Column(String(50), nullable=False)
    foto = Column(LONGTEXT, nullable=False)
    latitude = Column(String(50))
    longitude = Column(String(50))
    lokasi = Column(LONGTEXT)
    encoding = Column(BLOB)


# =========================
# FASTAPI Lifespan Event
# =========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        Base.metadata.create_all(bind=engine)
        logger.info(f"[STARTUP] Database siap: {engine.url}")
    except Exception:
        logger.exception("[STARTUP] Gagal inisialisasi database!")
        raise
    yield
    logger.info("[SHUTDOWN] Aplikasi ditutup.")


app = FastAPI(title="Face Verification Service", lifespan=lifespan)
bearer_scheme = HTTPBearer()


# =========================
# DEPENDENCY
# =========================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =========================
# FACE & IMAGE UTILITIES
# =========================
def get_image(bytes_data: bytes):
    image = Image.open(io.BytesIO(bytes_data))
    try:
        for key, val in ExifTags.TAGS.items():
            if val == "Orientation":
                orientation_key = key
                break
        exif = image._getexif()
        if exif and orientation_key in exif:
            if exif[orientation_key] == 3:
                image = image.rotate(180, expand=True)
            elif exif[orientation_key] == 6:
                image = image.rotate(270, expand=True)
            elif exif[orientation_key] == 8:
                image = image.rotate(90, expand=True)
    except Exception:
        pass

    if max(image.size) > 1024:
        image.thumbnail((1024, 1024))
    return image


def get_encodings(image):
    import face_recognition
    rgb = np.array(image.convert("RGB"))
    return face_recognition.face_encodings(rgb)


# =========================
# AUTH HELPER FUNCTIONS
# =========================
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db)
):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("email")
        if not email:
            raise HTTPException(401, "Token tidak valid!")
        user = db.query(User).filter_by(email=email).first()
        if not user:
            raise HTTPException(401, "User tidak ditemukan!")
        return user
    except JWTError:
        raise HTTPException(401, "Token tidak valid!")


# =========================
# ENDPOINTS
# =========================
@app.get("/")
def root():
    masked_url = str(engine.url).split("@")[0] + "@***"
    return {"ok": True, "service": "face-verification", "db": masked_url}


@app.post("/login/")
async def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=email).first()
    if not user or not verify_password(password, user.password):
        raise HTTPException(401, "Email atau password salah!")
    token = create_access_token({"email": user.email, "name": user.nama})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"id": user.id, "nama": user.nama, "email": user.email},
    }


@app.post("/register-scan/")
async def register_scan(
    file: UploadFile = File(...),
    nama: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    import face_recognition
    try:
        if db.query(User).filter_by(email=email).first():
            raise HTTPException(409, "Email sudah terdaftar!")
        if db.query(User).filter_by(nama=nama).first():
            raise HTTPException(409, "Nama sudah terdaftar!")

        data = await file.read()
        image = get_image(data)
        encs = get_encodings(image)
        if not encs:
            raise HTTPException(400, "Tidak ada wajah terdeteksi.")
        face_encoding = encs[0]

        # Cegah duplikasi wajah
        for u in db.query(User).all():
            if u.encoding:
                u_enc = pickle.loads(u.encoding)
                if isinstance(u_enc, list):
                    u_enc = u_enc[0]
                distance = face_recognition.face_distance([np.array(u_enc)], face_encoding)[0]
                if distance < THRESHOLD:
                    raise HTTPException(409, f"Wajah sudah terdaftar atas nama {u.nama}!")

        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        foto_base64 = base64.b64encode(data).decode('utf-8')
        user_id = str(uuid.uuid4())[:6]

        user = User(
            id=user_id,
            nama=nama,
            email=email,
            password=hashed_pw,
            foto_registrasi=foto_base64,
            encoding=pickle.dumps([face_encoding]),
        )
        db.add(user)
        db.commit()

        return {"status": "completed", "progress": 100, "instruction": "Registrasi selesai!", "user_id": user_id}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception("Register error")
        raise HTTPException(500, f"DB Error: {e}")


@app.post("/refresh/")
async def refresh_token(email: str = Body(...), db: Session = Depends(get_db)):
    user = db.query(User).filter_by(email=email).first()
    if not user:
        raise HTTPException(404, "User tidak ditemukan!")
    new_token = create_access_token({"email": user.email, "name": user.nama})
    return {"access_token": new_token, "token_type": "bearer"}


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
    import face_recognition
    try:
        now = datetime.now()
        today = now.date()
        jenis = aut.strip().lower()

        existing = db.query(Absen).filter(
            and_(Absen.id == user.id, Absen.date == today, Absen.aut.ilike(jenis))
        ).first()
        if existing:
            raise HTTPException(409, f"Anda sudah absen {aut} hari ini!")

        data = await file.read()
        image = get_image(data)
        encs = get_encodings(image)
        if not encs:
            raise HTTPException(400, "Tidak ada wajah terdeteksi.")
        target = encs[0]

        known = pickle.loads(user.encoding)
        known_encs = [np.array(e) for e in (known if isinstance(known, list) else [known])]
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
            encoding=pickle.dumps([target]),
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
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception("Absen error")
        raise HTTPException(500, f"Absen error: {e}")


@app.post("/verify/")
async def verify_face(
    file: UploadFile = File(...),
    nama: str = Form(...),
    db: Session = Depends(get_db)
):
    import face_recognition
    user = db.query(User).filter_by(nama=nama).first()
    if not user:
        raise HTTPException(404, f"Nama '{nama}' tidak ditemukan di database.")

    data = await file.read()
    image = get_image(data)
    encs = get_encodings(image)
    if not encs:
        raise HTTPException(400, "Tidak ada wajah terdeteksi.")
    target = encs[0]
    known = pickle.loads(user.encoding)
    distances = face_recognition.face_distance(
        [np.array(e) for e in (known if isinstance(known, list) else [known])], target
    )
    min_distance = float(np.min(distances))
    verified = bool(min_distance < THRESHOLD)
    return {"verified": verified, "nama": nama, "distance": min_distance, "threshold": THRESHOLD}


@app.get("/absen/history")
def get_absen_history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(Absen).filter(Absen.id == user.id).order_by(Absen.date.desc(), Absen.time.desc()).all()

    history = {}
    for ab in rows:
        date_str = ab.date.strftime("%A, %d %B %Y")
        jenis = ab.aut.lower()
        jam = ab.time
        lokasi = ab.lokasi

        if date_str not in history:
            history[date_str] = {"masuk": None, "keluar": None, "lokasi": lokasi}

        if "masuk" in jenis and not history[date_str]["masuk"]:
            history[date_str]["masuk"] = jam
        if "keluar" in jenis:
            history[date_str]["keluar"] = jam

    result = [{"tanggal": d, **v} for d, v in history.items()]
    result.sort(key=lambda x: datetime.strptime(x["tanggal"], "%A, %d %B %Y"), reverse=True)
    return {"history": result}


# =========================
# MAIN ENTRYPOINT
# =========================
if __name__ == "__main__":
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("✅ Koneksi database berhasil.")
    except Exception as e:
        logger.error(f"❌ Gagal konek ke database: {e}")

    module_name = os.path.splitext(os.path.basename(__file__))[0]
    logger.info(f"🚀 Server berjalan di http://127.0.0.1:{PORT}")
    uvicorn.run(f"{module_name}:app", host="127.0.0.1", port=PORT, reload=True)
