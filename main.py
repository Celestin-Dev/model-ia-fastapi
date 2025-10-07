import asyncio
from fastapi import FastAPI, Request, WebSocket, Depends
from fastapi.responses import JSONResponse
import numpy as np
from ultralytics import YOLO
import cv2
import time
from datetime import datetime
from util import get_car, read_license_plate, draw_border
from sort.sort import Sort
import base64
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from model import Base
from detection import *

# mémoire globale
detected_cars = {}
vehicle_positions = {}
vehicle_speeds = {}

def init_models():
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('license_plate_detector.pt')
    return coco_model, license_plate_detector

# traitement d'une frame
def process_frame(frame, coco_model, license_plate_detector, vehicles, mot_tracker,
                  previous_detections=None, vehicle_count=0):
    global detected_cars, vehicle_positions, vehicle_speeds

    if previous_detections is None:
        previous_detections = set()

    results = []
    detections = coco_model(frame)[0]
    detections_ = []
    current_detections = set()

    # Ligne de comptage (60% de la hauteur)
    height, width, _ = frame.shape
    line_position = int(height * 0.6)
    cv2.line(frame, (0, line_position), (width, line_position), (0, 0, 255), 2)

    # Dictionnaire classes COCO
    COCO_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    # Préparer les bbox pour SORT
    sort_input = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        y_center = int((y1 + y2) / 2)
        x_center = int((x1 + x2) / 2)

        # 🔹 Garder ta logique originale
        if int(class_id) in vehicles and y_center > line_position:
            current_detections.add(f"{x1}_{y1}_{x2}_{y2}")

            # Dessiner bordure et nouveau véhicule
            if f"{x1}_{y1}_{x2}_{y2}" not in previous_detections:
                vehicle_count += 1
                cv2.putText(frame, "NEW", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            draw_border(frame, (x1, y1), (x2, y2), (0, 255, 0), 25, line_length_x=200, line_length_y=200)

            # Ajouter bbox pour SORT
            sort_input.append([x1, y1, x2, y2, score])

    # 🔹 Tracker avec SORT
    track_ids = mot_tracker.update(np.asarray(sort_input))

    # 🔹 Calcul de la vitesse et affichage
    for i, track in enumerate(track_ids):
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2

        # Calcul vitesse stable
        speed_px = 0.0
        if track_id in vehicle_positions:
            prev_x, prev_y, prev_time = vehicle_positions[track_id]
            dt = time.time() - prev_time
            dx = x_center - prev_x
            dy = y_center - prev_y
            speed_px_new = ((dx**2 + dy**2)**0.5) / dt if dt > 0 else 0
            prev_speed = vehicle_speeds.get(track_id, 0.0)
            speed_px = 0.7 * prev_speed + 0.3 * speed_px_new
            vehicle_speeds[track_id] = speed_px

        vehicle_positions[track_id] = (x_center, y_center, time.time())

        # Conversion pixels/sec -> km/h
        pixel_to_meter = 0.01
        speed_kmh = speed_px * pixel_to_meter * 3.6

        # Récupérer classe
        class_id = int(detections.boxes.data[i][5])  # correspondance avec bbox
        vehicle_class_name = COCO_CLASSES.get(class_id, "unknown")

        # Affichage classe + vitesse
        cv2.putText(frame, f"{vehicle_class_name} {speed_kmh:.1f} km/h", (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    previous_detections.clear()
    previous_detections.update(current_detections)

    # 🔹 Détection plaques
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        center_y_plate = (y1 + y2) // 2
        if center_y_plate >= line_position:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            if car_id != -1:
                license_plate_crop = frame[y1:y2, x1:x2, :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                if license_plate_text is not None:
                    detected_cars[car_id] = {
                        'car_id': int(car_id),
                        'car_detection_score': float(score),
                        'car_bbox': [int(xcar1), int(ycar1), int(xcar2), int(ycar2)],
                        'license_plate_bbox': [x1, y1, x2, y2],
                        'license_number': license_plate_text,
                        'license_number_score': float(license_plate_text_score),
                        'speed_kmh': speed_kmh,
                        'vehicle_class': vehicle_class_name
                    }
                    results.append(detected_cars[car_id])

    # 🔹 Réaffichage des infos déjà connues
    for cid, car in detected_cars.items():
        x1, y1, x2, y2 = car['car_bbox']
        plate = car['license_number']
        vehicle_class_name = car.get('vehicle_class', 'unknown')
        cv2.putText(frame, f"ID:{cid} {vehicle_class_name} Plate:{plate}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Affichage compteur véhicules
    cv2.putText(frame, f'Nombre total de vehicules: {vehicle_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    return frame, results, vehicle_count




async def generate_detections(video_path="sample.mp4"):
    coco_model, license_plate_detector = init_models()
    mot_tracker = Sort()
    vehicles = [2, 3, 5, 7]  # car, motorbike, bus, truck
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    vehicle_count = 0
    previous_detections = set()
    frame_skip = 2  # traiter 1 frame sur 2

    frame_nmr = 0
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_nmr += 1
        if frame_nmr % frame_skip != 0:
            continue

        # YOLO peut traiter une version réduite si besoin
        # small_frame = cv2.resize(frame, (320, 180))
        # frame, detections, vehicle_count = process_frame(small_frame, ...)

        frame, detections, vehicle_count = process_frame(
            frame, coco_model, license_plate_detector, vehicles, mot_tracker,
            previous_detections, vehicle_count
        )

        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        frame_resized = cv2.resize(frame, (640, 360))
        _, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        frameb64 = base64.b64encode(buffer).decode("utf-8")

        yield {
            "frame_nmr": frame_nmr,
            "video": frameb64,
            "fps": fps,
            "detections": detections,
            "vehicle_count": vehicle_count,
            "all_detected_cars": list(detected_cars.values())
        }

        await asyncio.sleep(0)  # ne bloque pas l'event loop

    cap.release()



app = FastAPI()
Base.metadata.create_all(bind=engine)

# Dependency DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, db: Session = Depends(get_db)):
    await websocket.accept()
    async for result in generate_detections("sample.mp4"):
        for det in result["detections"]:
            save_detection(db, det["car_id"], det["car_detection_score"], det["license_number"], det["license_number_score"], det["car_bbox"], det["vehicle_class"], det["speed_kmh"])
        await websocket.send_json(result)


@app.get("/vehicle")
def read_vehicle_by_plate(license_plate: str, license_plate_score:float, timestamp:str, db: Session = Depends(get_db)):
    vehicle = get_vehicle_by_number_plate(db, license_plate, license_plate_score, timestamp)
    if not vehicle:
        return {"message": f"Véhicule avec la plaque {license_plate} introuvable"}
    return vehicle


@app.get("/vehicle/search/detail")
def read_details_vehicle(license_plate: str, date_start: str, date_end:str, db: Session = Depends(get_db)):
    details = get_info_vehicle_by_nplate_car_id_datedetection(db, license_plate, date_start ,date_end)
    if not details:
        return {"message": "Aucun détail trouvé pour ce véhicule."}
    return details


@app.get("/vehicles")
def read_all_vehicles(db: Session = Depends(get_db)):
    vehicles = getAllVehicles(db)
    return vehicles

@app.get("/vehicles/recent")
def read_recent_vehicles(limit: int = 5, db: Session = Depends(get_db)):
    vehicles = getLastVehicles(db, limit)
    return vehicles

@app.get("/vehicles/last5_per_car")
def read_last_five_per_car(db: Session = Depends(get_db)):
    return get_last_five_per_car(db)

@app.get("/vehicles/best_per_car")
def read_best_detection_per_car(db: Session = Depends(get_db)):
    return get_best_detection_per_car(db)


from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

class CameraStream(VideoStreamTrack):
    def __init__(self, source=0):
        super().__init__()
        self.cap = cv2.VideoCapture(source)

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        ret, frame = self.cap.read()
        if not ret:
            await asyncio.sleep(0.1)
            return None
        frame = cv2.resize(frame, (640, 360))
        av_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        av_frame.pts = pts
        av_frame.time_base = time_base
        return av_frame

# Route d’offre WebRTC
from aiortc.contrib.media import MediaPlayer
import os
pcs = set()

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    # --- choisir une source vidéo ---
    video_source = "sample.mp4"
    if not os.path.exists(video_source):
        video_source = 0  # utiliser webcam locale si pas de fichier

    player = MediaPlayer(video_source)

    if player.video:  # sécurise l’ajout
        pc.addTrack(player.video)
    if player.audio:
        pc.addTrack(player.audio)

    # --- WebRTC handshake ---
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    }

import uvicorn
if __name__== "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)