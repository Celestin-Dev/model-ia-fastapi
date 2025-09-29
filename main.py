import asyncio
from fastapi import FastAPI, WebSocket, Depends
import numpy as np
from ultralytics import YOLO
import cv2
import time
from util import get_car, read_license_plate, draw_border
from sort.sort import Sort
import base64
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from model import Base
from detection import save_detection, get_info_vehicle_by_nplate_car_id_datedetection, get_vehicle_by_number_plate
# mémoire globale des véhicules déjà détectés
detected_cars = {}
vehicle_positions = {}
vehicle_speeds = {}


def init_models():
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('license_plate_detector.pt')
    return coco_model, license_plate_detector

#traitement d'une frame
def process_frame(frame, coco_model, license_plate_detector, vehicles, mot_tracker, previous_detections=None, vehicle_count=0):
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

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        y_center = int((y1 + y2) / 2)
        x_center = int((x1 + x2)/2)

        if int(class_id) in vehicles and y_center > line_position:
            vehicle_id = f"{int(x1)}_{int(y1)}_{int(x2)}_{int(y2)}"
            current_detections.add(vehicle_id)
            if vehicle_id not in previous_detections:
                vehicle_count += 1
                cv2.putText(frame, "NEW", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            draw_border(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                        (0, 255, 0), 25, line_length_x=200, line_length_y=200)
            detections_.append([x1, y1, x2, y2, score])

            #calcul vitesse
            if(vehicle_id in vehicle_positions):
                prev_x, prev_y, prev_time = vehicle_positions[vehicle_id]
                dt = time.time() - prev_time
                dx = x_center - prev_x
                dy = y_center - prev_y
                speed_px = ((dx**2 + dy**2)**0.5) /dt if dt > 0 else 0
                vehicle_speeds[vehicle_id] = speed_px
            vehicle_positions[vehicle_id] = (x_center, y_center, time.time())

            #Affichage de vitesse
            if(vehicle_id in vehicle_speeds):
                speed_px = vehicle_speeds[vehicle_id]
                #conversion pixels/sec -> Km/h
                pixel_to_meter = 0.05
                speed_kmh = speed_px * pixel_to_meter * 3.6
                cv2.putText(frame, f"Speed: {speed_kmh:.2f} km/h", (int(x1), int(y1) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    previous_detections.clear()
    previous_detections.update(current_detections)

    # Tracking SORT
    track_ids = mot_tracker.update(np.asarray(detections_))

    # Détection plaques
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        center_y_plate = (y1 + y2) / 2
        if center_y_plate >= line_position:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            if car_id != -1:
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                if license_plate_text is not None:
                    # Sauvegarder les infos dans detected_cars
                    detected_cars[car_id] = {
                        'car_id': int(car_id),
                        'car_detection_score': float(score),
                        'car_bbox': [int(xcar1), int(ycar1), int(xcar2), int(ycar2)],
                        'license_plate_bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'license_number': license_plate_text,
                        'license_number_score': float(license_plate_text_score),
                        'speed_kmh': vehicle_speeds.get(f"{int(xcar1)}_{int(ycar1)}_{int(xcar2)}_{int(ycar2)}", 0)
                    }
                    results.append(detected_cars[car_id])

    # 🔹 Réafficher les infos déjà connues
    for cid, car in detected_cars.items():
        x1, y1, x2, y2 = car['car_bbox']
        plate = car['license_number']
        cv2.putText(frame, f"ID:{cid} Plate:{plate}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Affichage du compteur
    cv2.putText(frame, f'Nombre total de vehicules: {vehicle_count}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    return frame, results, vehicle_count


async def generate_detections(video_path="sample.mp4"):
    coco_model, license_plate_detector = init_models()
    mot_tracker = Sort()
    vehicles = [2, 3, 5, 7]  # COCO classes: car, motorbike, bus, truck
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    vehicle_count = 0
    previous_detections = set()

    frame_nmr = 0
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_nmr += 1
        frame, detections, vehicle_count = process_frame(
            frame, coco_model, license_plate_detector, vehicles, mot_tracker,
            previous_detections, vehicle_count
        )

        # calcul FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        # _, buffer = cv2.imencode(".jpg", frame)
        # frameb64 = base64.b64encode(buffer).decode("utf-8")
        # Réduire la taille + qualité JPEG plus faible
        frame_resized = cv2.resize(frame, (640, 360))
        _, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        frameb64 = base64.b64encode(buffer).decode("utf-8")


        yield {
            "frame_nmr": frame_nmr,
            "video": frameb64,
            "fps": fps,
            "detections": detections,          # résultats de cette frame
            "vehicle_count": vehicle_count,    # compteur incrémental
            "all_detected_cars": list(detected_cars.values())  # ✅ toutes les voitures mémorisées
        }

        await asyncio.sleep(0.03)  # pour ne pas bloquer l'event loop

    cap.release()


app = FastAPI()

Base.metadata.create_all(bind=engine)

#Dependency DB
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
        # Sauvegarder chaque détection dans MySQL
        for det in result["detections"]:
            save_detection(db, det["car_id"], det["car_detection_score"], det["license_number"], det["license_number_score"], det["car_bbox"])

        # Envoyer le flux vidéo + détections au client
        await websocket.send_json(result)

#recupere un vehicule par numero de plaque
@app.get("/vehicle/{license_plate}")
def read_vehicle_by_plate(license_plate: str, db: Session = Depends(get_db)):
    vehicle = get_vehicle_by_number_plate(db, license_plate)
    if not vehicle:
        return {"message": f"Véhicule avec la plaque {license_plate} introuvable"}
    return vehicle

#recupere les detailes de vehicule entre une date precis
@app.get("/vehicle/details")
def read_details_vehicle(
    license_plate: str,
    car_id: int,
    date_start: str,
    date_end: str,
    db: Session = Depends(get_db)
):
    details = get_info_vehicle_by_nplate_car_id_datedetection(
        db, license_plate, car_id, date_start, date_end
    )
    if not details:
        return {"message": "Aucun détail trouvé pour ce véhicule."}
    return details
