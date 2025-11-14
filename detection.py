import base64
from fastapi.responses import JSONResponse, Response
from sqlalchemy.orm import Session
from sqlalchemy import func, and_,extract 
from typing import Dict, Any, List, Optional
from model import CarDetection, Notification
from datetime import datetime, timedelta, date
import asyncio

def save_detection(db: Session, car_id: int, 
                   car_detection_score: float, 
                   license_plate: str, 
                   license_plate_score: float, 
                   bbox: list, car_class:str, 
                   car_speed: float, 
                   vehicle_color:str,
                   image_data: bytes = None,
                   video_id: str = "",
                   offense:bool=False
                   ):
    global location
    if video_id=="cam_001":
        location = "RN7-Entrée-Nord Fianarantsoa"
    elif video_id=="cam_002":
        location = "RN7-Sortie-Sud Fianarantsoa"


    license_plate_score = round(license_plate_score, 2)
    car_detection_score = round(car_detection_score, 2)
    detection = CarDetection(
        car_id=car_id,
        car_detection_score=car_detection_score,
        license_plate=license_plate,
        license_plate_score=license_plate_score,
        car_class=car_class,
        car_speed=car_speed,
        offense=offense,
        bbox_x=bbox[0],
        bbox_y=bbox[1],
        bbox_w=bbox[2]-bbox[0],
        bbox_h=bbox[3]-bbox[1],
        vehicle_color=vehicle_color,
        image_capture=image_data,
        video_id=video_id,
        location=location
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return detection

recent_notifs = {}

# Récupère les véhicules actuellement en infraction
async def save_notification(db: Session):
    results = db.query(CarDetection.car_id).filter(CarDetection.offense.is_(True)).all()
    return [{"car_id": r.car_id} for r in results]

# Surveille les nouvelles infractions et envoie + enregistre les notifications
from database import SessionLocal
async def monitor_offenses(notif_clients: list):
    last_check = datetime.utcnow() - timedelta(seconds=2)
    recent_notifs = {}

    while True:
        await asyncio.sleep(2)
        db = SessionLocal()
        try:
            new_offenses = db.query(CarDetection).filter(
                CarDetection.offense.is_(True),
                CarDetection.timestamp > last_check
            ).all()

            for offense in new_offenses:
                car_id = offense.car_id
                if car_id not in recent_notifs or datetime.utcnow() - recent_notifs[car_id] > timedelta(seconds=5):
                    notif = Notification(
                        car_id=car_id,
                        title="Dépassement",
                        is_read=False,
                        event_time=offense.timestamp
                    )
                    db.add(notif)
                    db.commit()
                    db.refresh(notif)
                    data = {
                        "car_id": car_id,
                        "title": notif.title,
                        "event_time": str(notif.event_time)
                    }

                    to_remove = []
                    for client in notif_clients:
                        try:
                            await client.send_json(data)
                        except Exception as e:
                            print("Erreur envoi notif:", e)
                            to_remove.append(client)
                    for client in to_remove:
                        notif_clients.remove(client)

                    recent_notifs[car_id] = datetime.utcnow()

            last_check = datetime.utcnow()
        finally:
            db.close()


# Recuperer les total de detection
def get_total_detections(db: Session) -> int:
    return db.query(CarDetection).count()

# Recuperer les detection aujourd'hui
def get_detections_today(db: Session) -> int:
    today = date.today()
    return db.query(CarDetection).filter(
        extract('year', CarDetection.timestamp) == today.year,
        extract('month', CarDetection.timestamp) == today.month,
        extract('day', CarDetection.timestamp) == today.day
    ).count()

# Recuperer les nombre de camera active aujourd'hui
def get_active_cameras_today(db: Session) -> int:
    today = date.today()
    active_ids = db.query(CarDetection.video_id).filter(
        extract('year', CarDetection.timestamp) == today.year,
        extract('month', CarDetection.timestamp) == today.month,
        extract('day', CarDetection.timestamp) == today.day
    ).distinct().all()
    return len(active_ids)

# Recuperer les total des infractions de vitesse
def get_total_offenses(db: Session) -> int:
    return db.query(CarDetection).filter(CarDetection.offense == True).count()


# Recuperer le recent detection
def get_recent_detections_best_score(db: Session, limit: int = 10):
    subquery = db.query(
        CarDetection.license_plate,
        func.max(CarDetection.timestamp).label("max_timestamp")
    ).group_by(CarDetection.license_plate).subquery()
    recent_unique_detections = db.query(CarDetection) \
        .join(
            subquery,
            (CarDetection.license_plate == subquery.c.license_plate) & 
            (CarDetection.timestamp == subquery.c.max_timestamp)
        ) \
        .filter(CarDetection.license_plate.isnot(None)) \
        .order_by(CarDetection.timestamp.desc()) \
        .limit(limit) \
        .all()
        
    return recent_unique_detections

# Rechercher par plaque d'immatriculation
def search_license_plate(db: Session, plate_query: str, limit: int = 20) -> List[CarDetection]:
    return db.query(CarDetection) \
             .filter(CarDetection.license_plate.ilike(f"%{plate_query}%")) \
             .order_by(CarDetection.timestamp.desc()) \
             .limit(limit) \
             .all()

# Recuper une image
def get_image_data_by_detection_id(db: Session, detection_id: int) -> bytes | None:
    vehicle = db.query(CarDetection).filter(CarDetection.id == detection_id).first()

    if not vehicle or not vehicle.image_capture:
        return JSONResponse(status_code=404, content={"detail": "Image non trouvée"})

    return Response(content=vehicle.image_capture, media_type="image/jpeg")

# Recuperer le recent infraction
def get_recent_offenses(db: Session, limit: int = 10):
    subquery = db.query(
        CarDetection.license_plate,
        func.max(CarDetection.timestamp).label("max_timestamp")
    ).filter(
        CarDetection.offense == True, 
        CarDetection.license_plate.isnot(None)
    ).group_by(CarDetection.license_plate).subquery()

    recent_unique_offenses = db.query(CarDetection) \
        .join(
            subquery,
            (CarDetection.license_plate == subquery.c.license_plate) & 
            (CarDetection.timestamp == subquery.c.max_timestamp)
        ) \
        .order_by(CarDetection.timestamp.desc()) \
        .limit(limit) \
        .all()
        
    return recent_unique_offenses

# Créer une notification d'infraction
def create_offense_notification(db: Session, car_det: CarDetection, offense_type: str):
    license_plate = car_det.license_plate
    if not license_plate:
        return None
    
    downtime_threshold = datetime.now() - timedelta(seconds=30)

    existing_notif = db.query(Notification) \
        .join(CarDetection, Notification.car_id == CarDetection.id) \
        .filter(
            CarDetection.license_plate == license_plate,
            Notification.event_time > downtime_threshold  
        ) \
        .first()
    if existing_notif:
        return None
    

    car_det_id = car_det.id 

    title = f"Infraction détectée:Plaque {license_plate}"

    new_notification = Notification(
        car_id=car_det_id,
        title=title,
        event_time=car_det.timestamp, 
        is_read=False
    )
    
    db.add(new_notification)
    db.commit()
    db.refresh(new_notification)
    
    ws_data = {
        "id": new_notification.id,
        "title": new_notification.title,
        "event_time": new_notification.event_time.isoformat(),
        "license_plate": car_det.license_plate,
        "location": car_det.location,
        "car_image_url": f"/api/images/{car_det.id}"
    }
    return ws_data


def get_car_detection_by_id(db: Session, car_id: int) -> CarDetection | None:
    car_detection = db.query(CarDetection) \
        .filter(CarDetection.car_id == car_id) \
        .order_by(CarDetection.timestamp.desc()) \
        .first()

    return car_detection


# Distribuer par chaque camera
def get_detection_volume_by_camera(db: Session, period: str = "today") -> List[Dict[str, Any]]:
    query = db.query(
        CarDetection.video_id, 
        func.count(CarDetection.id).label("total_detections")
    ).filter(CarDetection.video_id.isnot(None))
    
    
    if period == "today":
        today = date.today()
        query = query.filter(
            extract('year', CarDetection.timestamp) == today.year,
            extract('month', CarDetection.timestamp) == today.month,
            extract('day', CarDetection.timestamp) == today.day
        )
    
    elif period == "week":
        seven_days_ago = datetime.now() - timedelta(days=7)
        query = query.filter(CarDetection.timestamp >= seven_days_ago)
        
    elif period == "month":
        first_day_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        query = query.filter(CarDetection.timestamp >= first_day_of_month)
    
    camera_counts = query \
        .group_by(CarDetection.video_id) \
        .order_by(func.count(CarDetection.id).desc()) \
        .all()
    results = [
        {"camera_id": vid_id, "detection_count": count}
        for vid_id, count in camera_counts
    ]
    
    return results

# Distribution par classe de véhicule
def get_vehicle_class_distribution(db: Session, period: str = "today") -> List[Dict[str, Any]]:
    query = db.query(
        CarDetection.car_class, 
        func.count(CarDetection.id).label("count")
    ).filter(CarDetection.car_class.isnot(None))
    
    if period == "today":
        today = date.today()
        query = query.filter(
            extract('year', CarDetection.timestamp) == today.year,
            extract('month', CarDetection.timestamp) == today.month,
            extract('day', CarDetection.timestamp) == today.day
        )
    
    elif period == "week":
        seven_days_ago = datetime.now() - timedelta(days=7)
        query = query.filter(CarDetection.timestamp >= seven_days_ago)
        
    elif period == "month":
        first_day_of_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        query = query.filter(CarDetection.timestamp >= first_day_of_month)
        
    
    class_counts = query \
        .group_by(CarDetection.car_class) \
        .order_by(func.count(CarDetection.id).desc()) \
        .all()

    total_detections = sum(item.count for item in class_counts)
    
    if total_detections == 0:
        return []
        
    results = []
    for class_name, count in class_counts:
        results.append({
            "vehicle_class": class_name if class_name else "Inconnu",
            "count": count,
            "percentage": round((count / total_detections) * 100, 2)
        })
        
    results.sort(key=lambda x: x['count'], reverse=True)
    
    return results

from enum import Enum
class Period(str, Enum):
    TODAY = "today"
    WEEK = "week"
    MONTH = "month"

def apply_time_filter(query, period: Period):
    now = datetime.now()
    if period == Period.TODAY:
        query = query.filter(
            extract('year', CarDetection.timestamp) == now.year,
            extract('month', CarDetection.timestamp) == now.month,
            extract('day', CarDetection.timestamp) == now.day
        )
    
    elif period == Period.WEEK:
        seven_days_ago = now - timedelta(days=7)
        query = query.filter(CarDetection.timestamp >= seven_days_ago)
        
    elif period == Period.MONTH:
        first_day_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        query = query.filter(CarDetection.timestamp >= first_day_of_month)
        
    return query

def get_count_stats(db: Session, period: Period) -> Dict[str, Any]:
    
    base_query = db.query(CarDetection)
    filtered_query = apply_time_filter(base_query, period)
    total_detections = filtered_query.count()
    active_cameras = filtered_query.with_entities(CarDetection.video_id).distinct().count()
    vehicles_reported = filtered_query.filter(CarDetection.offense == 1).count()

    return {
        "total_detections": total_detections,
        "active_cameras": active_cameras,
        "vehicles_reported": vehicles_reported
    }

# Calcule le nombre de détections par jour pour la période spécifiée
def get_daily_detection_volume(db: Session, period: Period) -> List[Dict[str, Any]]:
    now = datetime.now()
    if period == Period.TODAY:
        start_date = now - timedelta(days=1) 
    elif period == Period.WEEK:
        start_date = now - timedelta(days=7) 
    elif period == Period.MONTH:
        start_date = now.replace(day=1) 

    results = db.query(
        func.date(CarDetection.timestamp).label("detection_day"), 
        func.count(CarDetection.id).label("detection_count")
    ).filter(
        CarDetection.timestamp >= start_date,
        CarDetection.timestamp < now
    ).group_by(
        func.date(CarDetection.timestamp)
    ).order_by(
        "detection_day"
    ).all()
    formatted_results = []
    for day_str, count in results:
        try:
            day_obj = datetime.strptime(str(day_str), '%Y-%m-%d')
            day_label = day_obj.strftime('%m/%d/%Y')
        except ValueError:
            day_label = str(day_str)
            
        formatted_results.append({
            "day_label": day_label,
            "count": count
        })
        
    return formatted_results


# Recuperer un vehicule par identification
def get_vehicle_by_id(db: Session, id: int)-> CarDetection | None:
    return db.query(CarDetection).filter(CarDetection.id == id).first()

