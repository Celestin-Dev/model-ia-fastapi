from sqlalchemy.orm import Session
from sqlalchemy import func
from model import CarDetection
from datetime import datetime, timedelta

def save_detection(db: Session, car_id: int, car_detection_score: float, license_plate: str, license_plate_score: float, bbox: list, car_class:str, car_speed):
    license_plate_score = round(license_plate_score, 2)
    car_detection_score = round(car_detection_score, 2)
    detection = CarDetection(
        car_id=car_id,
        car_detection_score=car_detection_score,
        license_plate=license_plate,
        license_plate_score=license_plate_score,
        car_class=car_class,
        car_speed=car_speed,
        bbox_x=bbox[0],
        bbox_y=bbox[1],
        bbox_w=bbox[2]-bbox[0],
        bbox_h=bbox[3]-bbox[1]
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return detection

def get_vehicle_by_number_plate(db: Session, license_plate: str, license_plate_score: float, timestamp:str):
    date = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

    start = date - timedelta(seconds=1)
    end = date + timedelta(seconds=1)
    epsilon = 0.01
    return db.query(CarDetection).filter(
        CarDetection.license_plate == license_plate,
        CarDetection.license_plate_score.between(license_plate_score - epsilon, license_plate_score + epsilon),
        CarDetection.timestamp.between(start, end)
    ).all()


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


def get_last_five_per_car(db: Session):
    # On ajoute un ROW_NUMBER() partitionné par car_id
    row_number = func.row_number().over(
        partition_by=CarDetection.car_id,
        order_by=CarDetection.timestamp.desc()
    ).label("rn")

    subquery = db.query(
        CarDetection.car_id,
        CarDetection.license_plate,
        CarDetection.car_detection_score,
        CarDetection.license_plate_score,
        CarDetection.timestamp,
        row_number
    ).subquery()

    # On garde uniquement les rn <= 5
    query = db.query(subquery).filter(subquery.c.rn <= 5).all()

    return [
        {
            "car_id": row.car_id,
            "license_plate": row.license_plate,
            "car_detection_score": row.car_detection_score,
            "license_plate_score": row.license_plate_score,
            "timestamp": row.timestamp
        }
        for row in query
    ]

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
            "timestamp": row.timestamp
        }
        for row in query
    ]