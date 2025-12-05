import asyncio
import json
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, Depends, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from io import BytesIO
import numpy as np
from sqlalchemy import desc, literal_column, and_
from torch import device
from ultralytics import YOLO
from typing import Set
import cv2
from datetime import datetime
import base64
import time
from contextlib import asynccontextmanager
from util import get_car, get_dominant_color, read_license_plate, draw_border
from sort.sort import Sort
from sqlalchemy.orm import Session
from database import SessionLocal, engine 
from model import Base
from detection import *
from concurrent.futures import ThreadPoolExecutor

video_clients: list[WebSocket] = []
notif_clients: list[WebSocket] = []
detection_queue: asyncio.Queue = asyncio.Queue()

# ThreadPool pour les tâches CPU lourdes
executor = ThreadPoolExecutor(max_workers=4)

# Initialisation des modèles
def init_models():
    coco_model = YOLO('yolov8n_int8_openvino_model', task="detect")
    license_plate_detector = YOLO('license_plate_detector_int8_openvino_model', task="detect")
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

# Stockage chaque vidéo
video_streams_state = {} 

def get_video_state(video_id):
    """Récupère ou initialise l'état pour une vidéo donnée."""
    if video_id not in video_streams_state:
        video_streams_state[video_id] = {
            "mot_tracker": Sort(),
            "detected_cars": {},
            "vehicle_positions": {},
            "vehicle_speeds": {},
            "captured_car_ids": set(),
        }
    return video_streams_state[video_id]


# Traitement d'une frame
def process_frame(frame, 
                  coco_model, 
                  license_plate_detector, 
                  vehicles,
                  video_id,
                  speed_limit):
    
    video_state = get_video_state(video_id)
    detected_cars = video_state["detected_cars"]
    vehicle_positions = video_state["vehicle_positions"]
    vehicle_speeds = video_state["vehicle_speeds"]
    captured_car_ids = video_state["captured_car_ids"]
    mot_tracker = video_state['mot_tracker']

    results = []
    
    # 1. Détection des véhicules
    detections_yolo = coco_model(frame, conf=0.3, classes=vehicles, device='cpu')[0]
    # detections_yolo = detections_yolo.cuda()


    height, width, _ = frame.shape
    line_position = int(height * 0.6)
    cv2.line(frame, (0, line_position), (width, line_position), (0, 0, 255), 2)

    COCO_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    sort_input = []
    bbox_to_class_id_map = {}

    for detection in detections_yolo.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        y_center = int((y1 + y2) / 2)

        # Filtrer par classe et position
        if int(class_id) in vehicles and y_center > line_position:
            sort_input.append([x1, y1, x2, y2, score])
            bbox_key=f"{x1}_{y1}_{x2}_{y2}"
            bbox_to_class_id_map[bbox_key] = (int(class_id), score)
            draw_border(frame, (x1, y1), (x2, y2), (0, 255, 0), 25, 200, 200)

    # 2. Tracker avec SORT
    track_ids = mot_tracker.update(np.asarray(sort_input))

    # 3. Traitement par véhicule
    for i, track in enumerate(track_ids):
        xcar1, ycar1, xcar2, ycar2, track_id = [int(v) for v in track[:5]]
        x_center = (xcar1 + xcar2) // 2
        y_center = (ycar1 + ycar2) // 2
        
        # Calcul vitesse
        speed_kmh = 0.0
        if track_id in vehicle_positions:
            prev_x, prev_y, prev_time = vehicle_positions[track_id]
            dt = time.time() - prev_time
            if dt > 0:
                dx = x_center - prev_x
                dy = y_center - prev_y
                speed_px_new = ((dx**2 + dy**2)**0.5) / dt
                prev_speed = vehicle_speeds.get(track_id, 0.0)
                speed_px = 0.7 * prev_speed + 0.3 * speed_px_new
                vehicle_speeds[track_id] = speed_px
                
                pixel_to_meter = 0.01
                speed_kmh = speed_px * pixel_to_meter * 3.6
        
        vehicle_positions[track_id] = (x_center, y_center, time.time())
        

        # Vérifier si on a déjà toutes les infos pour ce 'track_id'
        existing_data = detected_cars.get(track_id, {})
        vehicle_class_name = existing_data.get('vehicle_class')

        car_detection_score = existing_data.get('car_detection_score', float(track[4]))

        # Si la classe du véhicule n'est pas encore définie pour ce track_id
        if vehicle_class_name is None or vehicle_class_name == '' or vehicle_class_name == 'Inconnu':
            if i < len(sort_input):
                initial_bbox_data = sort_input[i]
                x1_init, y1_init, x2_init, y2_init = [int(v) for v in initial_bbox_data[:4]]
                
                initial_bbox_key = f"{x1_init}_{y1_init}_{x2_init}_{y2_init}"
                
                class_id_from_map = bbox_to_class_id_map.get(initial_bbox_key)
                
                if class_id_from_map is not None:
                    class_id_map, score_map = class_id_from_map
                    vehicle_class_name = COCO_CLASSES.get(class_id_map, 'Inconnu')
                    car_detection_score = score_map

                    # Mise en cache de la classe
                    detected_cars.setdefault(track_id, {})['vehicle_class'] = vehicle_class_name
                    detected_cars.setdefault(track_id,{})['car_detection_score'] = car_detection_score
                else:
                    vehicle_class_name = existing_data.get('vehicle_class', 'Inconnu')
        
        # 3. Mettre à jour la variable locale
        if vehicle_class_name not in ["","Inconnu"]:
            detected_cars.setdefault(track_id, {})['vehicle_class'] = vehicle_class_name


        if existing_data.get('license_number') and existing_data.get('vehicle_color') and vehicle_class_name not in ["", "Inconnu"]:
            existing_data['speed_kmh'] = speed_kmh
            detected_cars[track_id] = existing_data
            detected_cars[track_id] = existing_data
            results.append(existing_data.copy())
            continue 

        # 4. Exécuter la détection de plaque
        try:
            car_crop = frame[ycar1:ycar2, xcar1:xcar2, :]
            
            # Exécuter le détecteur de plaque sur le petit crop
            license_plates = license_plate_detector(car_crop, conf=0.5)[0]
            
            if len(license_plates.boxes.data) > 0:
                plate = license_plates.boxes.data.tolist()[0]
                xp1, yp1, xp2, yp2, plate_score, _ = plate
                xp1, yp1, xp2, yp2 = int(xp1), int(yp1), int(xp2), int(yp2)

                # 5.Exécuter l'OCR
                plate_crop = car_crop[yp1:yp2, xp1:xp2, :]
                plate_crop_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                _, plate_crop_thresh = cv2.threshold(plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
                
                license_plate_text, license_plate_text_score = read_license_plate(plate_crop_thresh)

                if license_plate_text:
                    
                    # Exécuter la détection de couleur
                    vehicle_color_name = existing_data.get('vehicle_color')
                    if vehicle_color_name is None:
                        vehicle_color_name = get_dominant_color(car_crop)
                    
                    # 6. Sauvegarder l'image
                    image_binary_data = existing_data.get('image_data')
                    if track_id not in captured_car_ids and image_binary_data is None:
                        ret, buffer = cv2.imencode('.jpg', car_crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        if ret:
                            image_binary_data = buffer.tobytes()
                            captured_car_ids.add(track_id)
                            
                    # 7. Gerer infraction
                    offense = False
                    if speed_kmh > speed_limit:
                        offense = True
                        

                    # 8. Stocker toutes les données
                    detected_cars[track_id] = {
                        'car_id': int(track_id),
                        'car_detection_score': car_detection_score,
                        'car_bbox': [xcar1, ycar1, xcar2, ycar2],
                        'license_plate_bbox': [xp1, yp1, xp2, yp2],
                        'license_number': license_plate_text,
                        'license_number_score': float(license_plate_text_score),
                        'speed_kmh': speed_kmh,
                        'vehicle_class': vehicle_class_name,
                        'vehicle_color': vehicle_color_name,
                        'image_data': image_binary_data,
                        'video_id': video_id,
                        'offense': offense
                    }
                    results.append(detected_cars[track_id])

        except Exception as e:
            print(f"Erreur de traitement pour track_id {track_id}: {e}")
            pass

    return frame, results

# Générateur de détections vidéo
async def generate_detections(video_id:str, video_path="sample.mp4"):
    video_state = get_video_state(video_id)

    vehicles = [2, 3, 5, 7]
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    frame_skip = 4
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
        
        # Initialisation de vitesse max
        SPEED_LIMIT = 0.4
        frame, detections = await asyncio.to_thread(process_frame,
            frame, coco_model, license_plate_detector, vehicles,
            video_id,SPEED_LIMIT
        )

        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        frame_resized = cv2.resize(frame, (640, 360))
        _, buffer = cv2.imencode('.jpg', frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        frameb64 = base64.b64encode(buffer).decode("utf-8")

        yield {
            "video_id": video_id,
            "frame_nmr": frame_nmr,
            "video": frameb64,
            "fps": fps,
            "detections": detections,
            "all_detected_cars": list(video_state['detected_cars'].values())
        }

        await asyncio.sleep(0)

    cap.release()


# Fonction Producteur de détections
async def detection_producer(video_id: str, video_path="sample.mp4"):
    try:
        # Itérer sur le générateur de détections
        async for result in generate_detections(video_id, video_path):
            db = SessionLocal()
            
            # Préparation des données pour la base de données et la diffusion
            result_to_broadcast = result.copy()
            serializable_detections = []
            try:
                for det in result.get("detections", []):
                    try:
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
                            det.get("image_data"),
                            video_id,
                            det.get('offense', False)
                        )
                        
                    except Exception as ex:
                        print("Erreur save_detection:", ex)
                        continue

                    car_det_instance = await asyncio.to_thread(get_car_detection_by_id, db, det["car_id"])
                    if car_det_instance and car_det_instance.offense:
                        try:
                            ws_data = await asyncio.to_thread(
                                create_offense_notification,
                                db,
                                car_det_instance,
                                car_det_instance.car_class 
                            )

                            # 3. Diffuser en temps réel la nouvelle notification
                            if ws_data:
                                # Encoder en JSON et diffuser
                                await broadcast_notification(json.dumps(ws_data)) 
                    
                        except Exception as ex:
                            print("Erreur de broadcast ou création notification:", ex)
                    
                    # Préparation pour la diffusion
                    det_copy = det.copy()
                    
                    if 'image_data' in det_copy:
                        del det_copy['image_data']
                        
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Démarrage du système de détection...")
    for video_id, video_path in VIDEO_STREAMS.items():
        asyncio.create_task(detection_producer(video_id, video_path))
        print(f"Lancement de la détection pour {video_id} ({video_path})")
    
    asyncio.create_task(detection_broadcaster())
    
    yield
    
    print("Arrêt du système de détection...")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# --------------- WebSocket endpoint for video -----------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

# ---------- WebSocket Endpoint pour le flux vidéo-------------------------
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



notif_clients: Set[WebSocket] = set()

# Fonction pour diffuser un message à tous les clients
async def broadcast_notification(message: str):
    send_tasks = []
    for client in list(notif_clients): 
        send_tasks.append(client.send_text(message))
    done, pending = await asyncio.wait(
        send_tasks,
        timeout=None,
        return_when=asyncio.FIRST_EXCEPTION,
    )
    for task in done:
        try:
            task.result()
        except WebSocketDisconnect:
            notif_clients.discard(task._input_w)
        except Exception as e:
            print(f"Erreur d'envoi WebSocket: {e}")



# --------------- WebSocket endpoint pour les notifications ----------------
@app.websocket("/ws/notifications")
async def websocket_notifications_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Ajouter le client à l'ensemble global
    notif_clients.add(websocket) 
    print(f"Nouveau client de notification connecté. Total: {len(notif_clients)}")

    try:
        while True:
            await websocket.receive_text() 
            
    except WebSocketDisconnect:
        notif_clients.discard(websocket)
        print(f"Client de notification déconnecté. Reste: {len(notif_clients)}")
    except Exception as e:
        print(f"Erreur inattendue sur WS notifications: {e}")
        notif_clients.discard(websocket)


# ----------------- Event startup --------------------
VIDEO_STREAMS = {
    "cam_001": "sample.mp4", 
    "cam_002": "sample.mp4",
}

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
#---------------------------- API endpoints---------------------------------
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

# Recuperation de statistique globales
@app.get("/dashboard/stats", response_model=Dict[str, Any])
def read_dashboard_stats(db: Session = Depends(get_db)):
    try:
        total_detections = get_total_detections(db)
        detections_today = get_detections_today(db)
        active_cameras = get_active_cameras_today(db)
        total_offenses = get_total_offenses(db)
        
        return {
            "total_detections": total_detections,
            "detections_today": detections_today,
            "active_cameras": active_cameras,
            "vehicles_reported": total_offenses,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des statistiques : {e}")

# Recuperer image
@app.get("/images/{detection_id}")
async def serve_car_image(detection_id: int, db: Session = Depends(get_db)):
    image_data = get_image_data_by_detection_id(db, detection_id)

    if image_data is None:
        raise HTTPException(status_code=404, detail="Image de détection non trouvée")
    
    image_buffer = BytesIO(image_data)
    
    return StreamingResponse(image_buffer, media_type="image/jpeg")

# Recuperer les vehicule recent
@app.get("/vehicles/recent", response_model=List[Dict[str, Any]])
def read_recent_detections(limit: int = 10, db: Session = Depends(get_db)):
    recent_cars = get_recent_detections_best_score(db, limit)
    results = []
    for car in recent_cars:
        timestamp_str = car.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        results.append({
            "id":car.id,
            "license_plate": car.license_plate,
            "license_plate_score": car.license_plate_score,
            "timestamp": timestamp_str,
            "video_id": car.video_id,
            "location": car.location if car.location else "Lieu inconnu",
            "image_capture": base64.b64encode(car.image_capture).decode("utf-8") if car.image_capture else None
        })
    return results

# Recherche par plaque
@app.get("/vehicles/search", response_model=List[Dict[str, Any]])
def search_vehicles(plate_query: str, db: Session = Depends(get_db)):
    if not plate_query:
        raise HTTPException(status_code=400, detail="Veuillez fournir un numéro de plaque à rechercher.")

    found_cars = search_license_plate(db, plate_query)

    results = []
    for car in found_cars:
        timestamp_str = car.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        results.append({
            "license_plate": car.license_plate,
            "detection_score": car.license_plate_score,
            "timestamp": timestamp_str,
            "camera_id": car.video_id,
            "location": car.location if car.location else "Lieu inconnu",
        })
    return results

# Recuperer les infraction recent
@app.get("/vehicles/offenses", response_model=List[Dict[str, Any]])
def read_recent_offenses(limit: int = 30, db: Session = Depends(get_db)):
    recent_offenses = get_recent_offenses(db, limit)
    results = []
    for offense in recent_offenses:
        timestamp_str = offense.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        results.append({
            "license_plate": offense.license_plate,
            "detection_score": offense.license_plate_score,
            "timestamp": timestamp_str,
            "camera_id": offense.video_id,
            "location": offense.location if offense.location else "Lieu inconnu",
            "image_capture": base64.b64encode(offense.image_capture).decode("utf-8") if offense.image_capture else None,
            "offense_type": offense.car_class
        })
    return results

# Lire les statistique de notification
@app.get("/notifications/stats", response_model=Dict[str, int])
def read_notification_stats(db: Session = Depends(get_db)):
    
    total_count = db.query(Notification).count()
    unread_count = db.query(Notification).filter(Notification.is_read == False).count()
    
    return {
        "total": total_count,
        "unread": unread_count
    }

# Lire l'historique des notifications
@app.get("/notifications/history", response_model=List[Dict[str, Any]])
def read_notification_history(db: Session = Depends(get_db), limit: int = 10):
    
    notifications = db.query(Notification).order_by(Notification.event_time.desc()).limit(limit).all()
    
    results = []
    for notif in notifications:
        results.append({
            "id": notif.id,
            "title": notif.title,
            "is_read": notif.is_read,
            "event_time": notif.event_time.isoformat(),
            "car_id": notif.car_id,
        })
    return results

# Marquer notification comme lue
@app.put("/notifications/mark_read/{notification_id}")
def mark_notification_as_read(notification_id: int, db: Session = Depends(get_db)): 
    notification = db.query(Notification).filter(Notification.id == notification_id).first()
    
    if not notification:
        raise HTTPException(status_code=404, detail="Notification non trouvée")
        
    notification.is_read = True
    db.commit()
    return {"message": f"Notification {notification_id} marquée comme lue"}

# Marquer toutes les notifications comme lues
@app.put("/notifications/mark_all_read")
def mark_all_notifications_as_read(db: Session = Depends(get_db)):
    db.query(Notification).filter(Notification.is_read == False).update(
        {Notification.is_read: True}, synchronize_session="fetch"
    )
    db.commit()
    return {"message": "Toutes les notifications non lues ont été marquées comme lues"}


# Distribuer par chaque camera
@app.get("/stats/distribution/camera", response_model=List[Dict[str, Any]])
def get_camera_distribution(
    db: Session = Depends(get_db),
    period: str = Query(default="today", regex="^(today|week|month)$") 
):
    try:
        camera_data = get_detection_volume_by_camera(db, period=period)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des données : {e}")

    total_all_cameras = sum(item["detection_count"] for item in camera_data)
    
    if total_all_cameras == 0:
        return []

    results_with_percent = []
    for item in camera_data:
        percent = round((item["detection_count"] / total_all_cameras) * 100, 2)
        item["percentage"] = percent
        results_with_percent.append(item)
        
    return results_with_percent


@app.get("/stats/distribution/vehicle_class", response_model=List[Dict[str, Any]])
def get_vehicle_class_distribution_api(
    db: Session = Depends(get_db),
    period: str = Query(default="today", regex="^(today|week|month)$") 
):
    return get_vehicle_class_distribution(db, period=period)


# Lire statistique
PeriodQuery = Query(
    Period.TODAY, 
    description="Période des statistiques. Valeurs: 'today', 'week', 'month'"
)
@app.get("/global/stats", response_model=Dict[str, Any])
async def read_global_stats(
    period: Period = PeriodQuery,
    db: Session = Depends(get_db)
):
    try:
        stats = get_count_stats(db, period)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {e}")

# Volume de Détection Quotidien
@app.get("/stats/volume/daily", response_model=List[Dict[str, Any]])
async def get_daily_detection_volume_api(
    period: Period = PeriodQuery,
    db: Session = Depends(get_db)
):
    try:
        return get_daily_detection_volume(db, period)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur : {e}")

# Vehicule by id
@app.get("/vehicle/by/id/{vehicle_id}")
def read_vehicle_by_id(vehicle_id: int, db: Session = Depends(get_db)):
    vehicle = get_vehicle_by_id(db, vehicle_id)
    if not vehicle:
        raise HTTPException(status_code=404, detail="Véhicule non trouvé")
    return {
        "license_plate": vehicle.license_plate,
        "detection_score": vehicle.license_plate_score,
        "timestamp": vehicle.timestamp.strftime("%Y-%m-%d %H:%M:%S") if vehicle.timestamp else None,
        "camera_id": vehicle.video_id,
        "location": vehicle.location or "Lieu inconnu",
        "image_capture": base64.b64encode(vehicle.image_capture).decode("utf-8") if vehicle.image_capture else None,
        "car_class": vehicle.car_class,
        "vehicle_color": vehicle.vehicle_color,
        "car_speed": vehicle.car_speed,
        "car_id": vehicle.car_id
    }


# Recherche de detection par divers critères
@app.get("/detections/search",
    summary="Rechercher des détections de véhicules",
    tags=["Detections"]
)
def search_car_detections(
    db: Session = Depends(get_db),
    license_plate: Optional[str] = Query(None, description="Plaque d'immatriculation (recherche partielle)."),
    car_class: Optional[str] = Query(None, description="Type de véhicule (ex: voiture)."),
    video_id: Optional[str] = Query(None, description="ID de la caméra source (ex: Camera 1)."),
    start_date: Optional[str] = Query(None, description="Date et heure de début (Format: 31-08-2025 11:23:00)"),
    end_date: Optional[str] = Query(None, description="Date et heure de fin (Format: 31-08-2025 11:23:00)"),
):
    start_dt = None
    end_dt = None
    DATE_FORMAT = "%d-%m-%Y %H:%M:%S"

    try:
        if start_date:
            start_dt = datetime.strptime(start_date, DATE_FORMAT)
        if end_date:
            end_dt = datetime.strptime(end_date, DATE_FORMAT)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Format de date/heure invalide. Attendu: {DATE_FORMAT}. Erreur: {e}"
        )

    query = db.query(CarDetection).distinct(CarDetection.car_id)
    filters = []

    if license_plate:
        filters.append(CarDetection.license_plate.ilike(f"%{license_plate}%"))

    if car_class:
        filters.append(CarDetection.car_class == car_class)
    if video_id:
        filters.append(CarDetection.video_id == video_id)
        
    if start_dt:
        filters.append(CarDetection.timestamp >= start_dt)

    if end_dt:
        filters.append(CarDetection.timestamp <= end_dt)

    if filters:
        query = query.filter(and_(*filters))
        
    window_function = func.row_number().over(
        partition_by=CarDetection.car_id,
        order_by=desc(CarDetection.timestamp)
    ).label("row_number")
    
    subquery = db.query(CarDetection, window_function).filter(and_(*filters)).subquery()
    query = db.query(subquery).filter(literal_column("row_number") == 1)
        
    results = query.order_by(desc(subquery.c.timestamp)).limit(10).all()
    
    return [
        {
            "id": det.id,
            "license_plate": det.license_plate,
            "detection_score": det.license_plate_score,
            "timestamp": det.timestamp.strftime("%Y-%m-%d %H:%M:%S") if det.timestamp else None,
            "camera_id": det.video_id,
            "location": det.location or "Lieu inconnu",
            "car_class": det.car_class,
            "vehicle_color": det.vehicle_color,
            "car_speed": det.car_speed,
            "car_id": det.car_id,
            "image_capture": base64.b64encode(det.image_capture).decode("utf-8") if det.image_capture else None
        }
        for det in results
    ]


# Reserche de detection entre deux date
@app.get(
    "/detections/by-date-range",
    summary="Récupérer toutes les détections entre une date de début et une date de fin",
    tags=["Detections"]
)
def get_detections_by_date_range(
    db: Session = Depends(get_db),
    date_start: str = Query(..., description="Date et heure de début (Format: JJ-MM-AAAA HH:MM:SS)"),
    date_end: str = Query(..., description="Date et heure de fin (Format: JJ-MM-AAAA HH:MM:SS)"),
):
    start_dt = None
    end_dt = None
    DATE_FORMAT = "%d-%m-%Y %H:%M:%S"
    try:
        start_dt = datetime.strptime(date_start, DATE_FORMAT)
        end_dt = datetime.strptime(date_end, DATE_FORMAT)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Format de date/heure invalide. Le format attendu est: {DATE_FORMAT}. Veuillez vérifier les valeurs fournies."
        )

    query = db.query(CarDetection).filter(
        and_(
            CarDetection.timestamp >= start_dt,
            CarDetection.timestamp <= end_dt
        )
    )
        
    results = query.order_by(desc(CarDetection.timestamp)).limit(1000).all()
    
    return [
        {
            "id": det.id,
            "license_plate": det.license_plate,
            "detection_score": det.license_plate_score,
            "timestamp": det.timestamp.strftime("%Y-%m-%d %H:%M:%S") if det.timestamp else None,
            "camera_id": det.video_id,
            "location": det.location or "Lieu inconnu",
            "car_class": det.car_class,
            "vehicle_color": det.vehicle_color,
            "car_speed": det.car_speed,
            "car_id": det.car_id,
            "image_capture": base64.b64encode(det.image_capture).decode("utf-8") if det.image_capture else None
        }
        for det in results
    ]

class DateFilter(str, Enum):
    ALL = "ALL"
    TODAY = "TODAY"
    WEEK = "WEEK"
    MONTH = "MONTH"

class CameraFilter(str, Enum):
    ALL_CAM = "ALL_CAM"
    CAM1 = "cam_001"
    CAM2 = "cam_002"

class TypeFilter(str, Enum):
    ALL_TYPE = "ALL_TYPE"
    CAR = "car"
    MOTO = "motorcycle"
    BUS = "bus"
    TRUCK = "truck"


@app.get("/detections/filter")
def get_filtered_detections(
    det_filter: DateFilter = Query(default=DateFilter.ALL),
    camera_filter: CameraFilter = Query(default=CameraFilter.ALL_CAM),
    type_filter: TypeFilter = Query(default=TypeFilter.ALL_TYPE),
    limit: int = Query(default=10, ge=1, le=100),
    skip: int = Query(default=0, ge=0),
    db: Session = Depends(get_db)
):
    # Appel de la fonction logique
    results_filter = filter_detections(
        db=db, 
        det_filter=det_filter.value, # .value pour récupérer la string de l'Enum
        camera_filter=camera_filter, 
        type_filter=type_filter,
        limite=limit,
        skip=skip
    )

    results = []

    for car in results_filter:
        
        results.append({
            "id":car.id,
            "license_plate": car.license_plate,
            "license_plate_score": car.license_plate_score,
            "timestamp": car.timestamp.strftime("%Y-%m-%d %H:%M:%S") if car.timestamp else None,
            "video_id": car.video_id,
            "location": car.location if car.location else "Lieu inconnu",
            "image_capture": base64.b64encode(car.image_capture).decode("utf-8") if car.image_capture else None
        })

    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)