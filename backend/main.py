from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.database import engine
from backend import models
from backend.auth import (
    get_db,
    hash_password,
    verify_password,
    create_token,
    get_current_user
)

# ================= DB INIT =================
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI-Based Smart Surveillance Backend")

# ================= ROOT =================
@app.get("/")
def root():
    return {"status": "Backend running"}

# ================= AUTH =================
@app.post("/register")
def register(
    username: str,
    password: str,
    db: Session = Depends(get_db)
):
    existing = db.query(models.User).filter(
        models.User.username == username
    ).first()

    if existing:
        raise HTTPException(status_code=400, detail="User already exists")

    user = models.User(
        username=username,
        password=hash_password(password)
    )
    db.add(user)
    db.commit()
    return {"status": "User registered successfully"}

@app.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(models.User).filter(
        models.User.username == form_data.username
    ).first()

    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_token({"sub": user.username})
    return {
        "access_token": token,
        "token_type": "bearer"
    }

@app.get("/protected")
def protected(user=Depends(get_current_user)):
    return {"message": f"Hello {user.username}"}

# ================= CAMERA REGISTRY =================
@app.post("/cameras")
def add_camera(
    name: str,
    source: str,
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    cam = models.Camera(name=name, source=source)
    db.add(cam)
    db.commit()
    return {"status": "Camera added"}

@app.get("/cameras")
def list_cameras(
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    return db.query(models.Camera).all()

@app.delete("/cameras/{camera_id}")
def delete_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user)
):
    cam = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
    if not cam:
        return {"error": "Camera not found"}
    db.delete(cam)
    db.commit()
    return {"status": "Camera removed"}
