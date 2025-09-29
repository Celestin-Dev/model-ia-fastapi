from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from database import Base

class CarDetection(Base):
    __tablename__ = "car_detections"

    id = Column(Integer, primary_key=True, index=True)
    car_id = Column(Integer, index=True)
    car_detection_score = Column(Float)
    license_plate = Column(String(50), nullable=True)
    license_plate_score = Column(Float)
    bbox_x = Column(Float)
    bbox_y = Column(Float)
    bbox_w = Column(Float)
    bbox_h = Column(Float)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
