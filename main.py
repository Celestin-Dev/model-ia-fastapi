import asyncio
from fastapi import FastAPI, Request, WebSocket, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from ultralytics import YOLO
import cv2
import time
from datetime import datetime
import base64

from util import get_car, get_dominant_color, read_license_plate, draw_border
from sort.sort import Sort  # Assurez-vous du bon chemin d'import
from sqlalchemy.orm import Session
from database import SessionLocal, engine   # Assurez-vous du bon chemin d'import
from model import Base
from detection import *
from concurrent.futures import ThreadPoolExecutor

# Mémoire globale
detected_cars = {}
vehicle_positions = {}
vehicle_speeds = {}
captured_car_ids = set()

video_clients: list[WebSocket] = []
notif_clients: list[WebSocket] = []
detection_queue: asyncio.Queue = asyncio.Queue()

# ThreadPool pour les tâches CPU lourdes
executor = ThreadPoolExecutor(max_workers=4)

# Initialisation des modèles
def init_models():
    coco_model = YOLO('yolov8n_int8_openvino_model')
    license_plate_detector = YOLO('license_plate_detector_int8_openvino_model')
    return coco_model, license_plate_detector

coco_model, license_plate_detector = init_models()

# Création des tables
Base.metadata.create_all(bind=engine)

# Dependency DB
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Traitement d'une frame
def process_frame(frame, coco_model, license_plate_detector, vehicles, mot_tracker,
                  previous_detections=None, vehicle_count=0):
    global detected_cars, vehicle_positions, vehicle_speeds, captured_car_ids

    if previous_detections is None:
        previous_detections = set()

    results = []
    detections = coco_model(frame)[0]
    current_detections = set()

    height, width, _ = frame.shape
    line_position = int(height * 0.6)
    cv2.line(frame, (0, line_position), (width, line_position), (0, 0, 255), 2)

    COCO_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    sort_input = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        y_center = int((y1 + y2) / 2)
        x_center = int((x1 + x2) / 2)

        if int(class_id) in vehicles and y_center > line_position:
            current_detections.add(f"{x1}_{y1}_{x2}_{y2}")

            if f"{x1}_{y1}_{x2}_{y2}" not in previous_detections:
                vehicle_count += 1
            draw_border(frame, (x1, y1), (x2, y2), (0, 255, 0), 25, 200, 200)

            sort_input.append([x1, y1, x2, y2, score])

    # Tracker avec SORT
    track_ids = mot_tracker.update(np.asarray(sort_input))

    # Vitesse et affichage
    for i, track in enumerate(track_ids):
        x1, y1, x2, y2, track_id = [int(v) for v in track[:5]]
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2

        # Calcul vitesse
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

        pixel_to_meter = 0.01
        speed_kmh = speed_px * pixel_to_meter * 3.6

        class_id = int(detections.boxes.data[i][5])
        vehicle_class_name = COCO_CLASSES.get(class_id, "unknown")

    previous_detections.clear()
    previous_detections.update(current_detections)

    # Détection plaques
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
                image_binary_data = None

                vehicle_color_name = "Inconnu" # Initialiser la couleur
                car_image_crop = None  # Initialiser le crop de la voiture

                if license_plate_text is not None:

                    existing_data = detected_cars.get(car_id, {})
                    vehicle_color_name = existing_data.get('vehicle_color')

                    # ---BLOC DE DÉTECTION DE COULEUR---
                    if vehicle_color_name is None:
                        try:
                            car_image_crop = frame[int(ycar1):int(ycar2), int(xcar1):int(xcar2), :]
                            vehicle_color_name = get_dominant_color(car_image_crop)
                            print(f"CarID {car_id}: Couleur détectée = {vehicle_color_name}")

                        except Exception as e:
                            print(f"Erreur lors du crop/détection de couleur: {e}")
                            vehicle_color_name = "Inconnu"
                        # ---FIN DU BLOC COULEUR---
                    else:
                        print(f"CarID {car_id}: Couleur détectée = {vehicle_color_name}")

                    if car_id not in captured_car_ids:
                        try:
                            if car_image_crop is None:
                                car_image_crop = frame[int(ycar1):int(ycar2), int(xcar1):int(xcar2), :]

                            ret, buffer = cv2.imencode('.jpg', car_image_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                            if ret:
                                image_binary_data = buffer.tobytes()
                                captured_car_ids.add(car_id)
                                print(f"Image encodée en mémoire pour car_id {car_id} ({len(image_binary_data)} octets)")
                            else:
                                print(f"Échec de l'encodage JPEG pour car_id {car_id}")

                        except Exception as e:
                            print(f"Erreur d'encodage image : {e}")
                            image_binary_data = None
                    else:
                        image_binary_data = detected_cars.get(car_id, {}).get('image_data')

                    detected_cars[car_id] = {
                        'car_id': int(car_id),
                        'car_detection_score': float(score),
                        'car_bbox': [int(xcar1), int(ycar1), int(xcar2), int(ycar2)],
                        'license_plate_bbox': [x1, y1, x2, y2],
                        'license_number': license_plate_text,
                        'license_number_score': float(license_plate_text_score),
                        'speed_kmh': speed_kmh,
                        'vehicle_class': vehicle_class_name,
                        'vehicle_color': vehicle_color_name,
                        'image_data': image_binary_data
                    }
                    results.append(detected_cars[car_id])

    return frame, results, vehicle_count

# Générateur de détections vidéo
async def generate_detections(video_path="sample.mp4"):
    mot_tracker = Sort()
    vehicles = [2, 3, 5, 7]
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    vehicle_count = 0
    previous_detections = set()
    frame_skip = 3
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

        frame, detections, vehicle_count = await asyncio.to_thread(process_frame,
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

        await asyncio.sleep(0)

    cap.release()

# Application FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# Fonction Producteur de détections
# Fonction Producteur de détections
async def detection_producer(video_path="sample.mp4"):
    """
    Génère des détections, sauvegarde les données brutes dans la DB/mémoire,
    et place une copie sérialisable dans la file d'attente de diffusion.
    """
    try:
        # Itérer sur le générateur de détections
        async for result in generate_detections(video_path):
            
            db = SessionLocal()
            
            # Préparation des données pour la base de données et la diffusion
            result_to_broadcast = result.copy()
            serializable_detections = []
            
            try:
                # Traiter les détections (Sauvegarde et création de la copie à diffuser)
                for det in result.get("detections", []):
                    
                    # --- Sauvegarde DB ---
                    try:
                        # Assurez-vous que 'save_detection' gère la donnée binaire correctement.
                        await asyncio.to_thread(
                            save_detection,
                            db,
                            det["car_id"],
                            det["car_detection_score"],
                            det.get("license_number"),
                            det.get("license_number_score", 0.0),
                            det.get("car_bbox"),
                            det.get("vehicle_class"),
                            det.get("speed_kmh", 0.0),
                            det.get("vehicle_color", "Inconnu"),
                            det.get("image_data") # Passe les octets à la DB
                        )
                    except Exception as ex:
                        print("Erreur save_detection:", ex)
                    
                    # --- Préparation pour la diffusion ---
                    det_copy = det.copy()
                    
                    if 'image_data' in det_copy:
                        del det_copy['image_data'] # Nettoyage N°1
                        
                    serializable_detections.append(det_copy)

                # Mettre à jour la copie du résultat avec les détections propres
                result_to_broadcast["detections"] = serializable_detections

                if "all_detected_cars" in result_to_broadcast:
                    del result_to_broadcast["all_detected_cars"]
                # Mettre la COPIE 100% propre dans la file d'attente
                await detection_queue.put(result_to_broadcast)
                
            finally:
                db.close()
                
    except Exception as e:
        print(f"Erreur fatale dans detection_producer: {e}")
        
    finally:
        print("detection_producer terminé")

# Background broadcaster
async def detection_broadcaster():
    while True:
        result = await detection_queue.get()
        to_remove = []
        for ws in video_clients:
            try:
                await ws.send_json(result)
            except Exception:
                to_remove.append(ws)
        for ws in to_remove:
            if ws in video_clients:
                video_clients.remove(ws)
        detection_queue.task_done()

# WebSocket endpoint for video
@app.websocket("/ws")
async def websocket_video(websocket: WebSocket):
    await websocket.accept()
    video_clients.append(websocket)
    print("Client vidéo connecté, total:", len(video_clients))
    try:
        while True:
            await asyncio.sleep(60)
    except Exception as e:
        print("websocket_video erreur:", e)
    finally:
        if websocket in video_clients:
            video_clients.remove(websocket)
        await websocket.close()
        print("Client vidéo déconnecté")

# WebSocket endpoint for notifications
@app.websocket("/ws/notifications")
async def websocket_notifications(websocket: WebSocket):
    await websocket.accept()
    notif_clients.append(websocket)
    print("Client notif connecté, total:", len(notif_clients))
    try:
        while True:
            await asyncio.sleep(60)
    except Exception as e:
        print("websocket_notifications erreur:", e)
    finally:
        if websocket in notif_clients:
            notif_clients.remove(websocket)
        await websocket.close()
        print("Client notif déconnecté")

# Event startup : lancer tasks background
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(detection_producer("sample.mp4"))
    asyncio.create_task(detection_broadcaster())
    # monitor_offenses doit être async et existant (sinon à commenter)
    # asyncio.create_task(monitor_offenses(notif_clients))

# API endpoints
@app.get("/save")
async def save_notif(db: Session = Depends(get_db)):
    return await save_notification(db)

@app.get("/vehicle")
def read_vehicle_by_plate(license_plate: str, license_plate_score: float, timestamp: str, db: Session = Depends(get_db)):
    vehicle = get_vehicle_by_number_plate(db, license_plate, license_plate_score, timestamp)
    if not vehicle:
        return {"message": f"Véhicule avec la plaque {license_plate} introuvable"}
    return vehicle

@app.get("/vehicle/search/detail")
def read_details_vehicle(license_plate: str, date_start: str, date_end: str, db: Session = Depends(get_db)):
    details = get_info_vehicle_by_nplate_car_id_datedetection(db, license_plate, date_start, date_end)
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

@app.get("/vehicles/best_per_car")
def read_best_detection_per_car(db: Session = Depends(get_db)):
    return get_best_detection_per_car(db)

@app.get("/notifications")
def read_notifications(db: Session = Depends(get_db)):
    return get_all_notifications(db)


@app.get("/cars/total_unique")
def read_total_unique_vehicles(db: Session = Depends(get_db)):
    return get_total_unique_vehicles(db)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
