import base64
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from model import CarDetection, Notification
from datetime import datetime, timedelta
import asyncio

def save_detection(db: Session, car_id: int, 
                   car_detection_score: float, 
                   license_plate: str, 
                   license_plate_score: float, 
                   bbox: list, car_class:str, 
                   car_speed: float, 
                   vehicle_color:str,
                   image_data: bytes = None,
                   ):
    license_plate_score = round(license_plate_score, 2)
    car_detection_score = round(car_detection_score, 2)
    detection = CarDetection(
        car_id=car_id,
        car_detection_score=car_detection_score,
        license_plate=license_plate,
        license_plate_score=license_plate_score,
        car_class=car_class,
        car_speed=car_speed,
        offense=car_speed > 0.30,
        bbox_x=bbox[0],
        bbox_y=bbox[1],
        bbox_w=bbox[2]-bbox[0],
        bbox_h=bbox[3]-bbox[1],
        vehicle_color=vehicle_color,
        image_capture=image_data
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



def get_all_notifications(db: Session):
    query = db.query(Notification).order_by(Notification.event_time.desc()).all()
    return [
        {
            "id": row.id,
            "car_id": row.car_id,
            "title": row.title,
            "is_read": row.is_read,
            "event_time": row.event_time
        }
        for row in query
    ]


def get_vehicle_by_number_plate(db: Session, license_plate: str, license_plate_score: float, timestamp:str):
    date = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

    start = date - timedelta(seconds=1)
    end = date + timedelta(seconds=1)
    epsilon = 0.01
    return db.query(CarDetection).filter(
        CarDetection.license_plate == license_plate,
        CarDetection.license_plate_score.between(license_plate_score - epsilon, license_plate_score + epsilon),
        CarDetection.timestamp.between(start, end)
    ).first()


def get_info_vehicle_by_nplate_car_id_datedetection(db: Session, license_plate: str, date_start: str, date_end:str):
    try:
        start = datetime.strptime(date_start, "%Y-%m-%d %H:%M:%S")
        end = datetime.strptime(date_end, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    epsilon = 0.01
    query = db.query(
        CarDetection.license_plate,
        CarDetection.license_plate_score,
        CarDetection.timestamp
        ).filter(
        CarDetection.license_plate == license_plate,
        CarDetection.timestamp.between(start, end)
    ).all()
    return [
        {
            "license_plate": row.license_plate,
            "license_plate_score": row.license_plate_score,
            "timestamp": row.timestamp
        }
        for row in query
    ]



def getAllVehicles(db: Session):
    query = db.query(
        CarDetection.license_plate, 
        CarDetection.timestamp
        ).all()
    return [
        {
            "license_plate": row.license_plate,
            "timestamp": row.timestamp
        }
        for row in query
    ]

def getLastVehicles(db: Session, limit: int = 5):
    query = db.query(
        CarDetection.license_plate,
        CarDetection.timestamp
    ).order_by(CarDetection.timestamp.desc()).limit(limit).all()

    return [
        {
            "license_plate": row.license_plate,
            "timestamp": row.timestamp
        }
        for row in query
    ]


def get_stat_vehicle(db:Session):
    data = db.query(CarDetection).filter(CarDetection.license_plate_score)

def get_best_detection_per_car(db: Session):

    # On récupère le maximum de license_plate_score pour chaque car_id
    subquery = db.query(
        CarDetection.car_id,
        func.max(CarDetection.license_plate_score).label("max_score")
    ).group_by(CarDetection.car_id).subquery()

    # On joint avec la table principale pour récupérer toutes les infos correspondantes
    query = db.query(CarDetection).join(
        subquery,
        (CarDetection.car_id == subquery.c.car_id) &
        (CarDetection.license_plate_score == subquery.c.max_score)
    ).all()

    # Transforme en dictionnaire pour FastAPI
    return [
        {
            "license_plate": row.license_plate,
            "license_plate_score": row.license_plate_score,
            "timestamp": row.timestamp,
            "image_capture": base64.b64encode(row.image_capture).decode("utf-8") if row.image_capture else None
        }
        for row in query
    ]


def get_last_overspeed_detections(db: Session, limit_speed: float = 80.0):
    subquery = (
        db.query(
            CarDetection.car_id,
            func.max(CarDetection.timestamp).label("latest_time")
        )
        .filter(CarDetection.car_speed > limit_speed)
        .group_by(CarDetection.car_id)
        .subquery()
    )

    result = (
        db.query(CarDetection)
        .join(subquery, (CarDetection.car_id == subquery.c.car_id) &
                        (CarDetection.timestamp == subquery.c.latest_time))
        .all()
    )

    return result


# Get total number of detections
def get_total_unique_vehicles(db: Session):
    latest_subq = (
        db.query(
            CarDetection.car_id.label("car_id"),
            func.max(CarDetection.timestamp).label("max_ts")
        )
        .group_by(CarDetection.car_id)
        .subquery()
    )

    # On joint la table principale avec la sous-requête pour ne garder que la ligne "dernière" par car_id
    latest_records_q = (
        db.query(CarDetection)
        .join(
            latest_subq,
            and_(
                CarDetection.car_id == latest_subq.c.car_id,
                CarDetection.timestamp == latest_subq.c.max_ts
            )
        )
        .subquery()  # représente l'ensemble des enregistrements représentatifs (une ligne par car_id)
    )

    # Maintenant, compter par class à partir de ces lignes représentatives
    results = (
        db.query(
            latest_records_q.c.car_class,
            func.count(latest_records_q.c.car_id).label("unique_vehicle_count")
        )
        .group_by(latest_records_q.c.car_class)
        .all()
    )

    # Total global = nombre de lignes dans latest_records_q (une ligne par car_id)
    total_unique = db.query(func.count(latest_records_q.c.car_id)).scalar()

    return {
        "total_unique_vehicles": total_unique,
        "details_by_class": [
            {"car_class": car_class, "unique_vehicle_count": count}
            for car_class, count in results
        ]
    }