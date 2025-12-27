# AI-Based Multi-Camera Person & Object Tracking System (Simulation)

## 📌 Overview
This project is an AI-powered surveillance system that detects, tracks, and identifies people and objects across multiple camera feeds.  
It is built as a **simulation** for academic and internship purposes.

## 🎯 Features
- Multi-camera support (video files, webcam, mobile IP camera)
- Person & object detection using YOLOv8
- Face recognition (Known / Unknown)
- Secure backend with JWT authentication
- Camera add / list / delete APIs
- Scalable architecture

## 🛠️ Tech Stack
**AI / CV**
- Python
- OpenCV
- YOLOv8
- face_recognition

**Backend**
- FastAPI
- JWT Authentication
- SQLite
- SQLAlchemy

**Frontend (Planned)**
- React
- Axios

## 🚀 How to Run

### Backend
```bash
uvicorn backend.main:app --reload
