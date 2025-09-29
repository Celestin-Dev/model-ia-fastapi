from sqlalchemy.orm import Session
from model import CarDetection
from datetime import datetime

def save_detection(db: Session, car_id: int, car_detection_score: float, license_plate: str, license_plate_score: float, bbox: list):
    detection = CarDetection(
        car_id=car_id,
        car_detection_score=car_detection_score,
        license_plate=license_plate,
        license_plate_score=license_plate_score,
        bbox_x=bbox[0],
        bbox_y=bbox[1],
        bbox_w=bbox[2]-bbox[0],
        bbox_h=bbox[3]-bbox[1]
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return detection

#Recupere les vehicule qui porte le numero que vous demande
def get_vehicle_by_number_plate(db: Session, license_plate: str):
    return db.query(CarDetection).filter(CarDetection.license_plate == license_plate).all()

#Recupere les details de vehicules par rapport a la plaque, l'id et la date
def get_info_vehicle_by_nplate_car_id_datedetection(
    db: Session,
    license_plate: str,
    car_id: int,
    date_start: str,
    date_end: str
):
    # Convertir les chaînes en datetime
    try:
        start = datetime.fromisoformat(date_start)
        end = datetime.fromisoformat(date_end)
    except ValueError:
        return None  # dates invalides

    query = db.query(CarDetection).filter(
        CarDetection.license_plate == license_plate,
        CarDetection.car_id == car_id,
        CarDetection.timestamp.between(start, end)
    )

    return query.all()
