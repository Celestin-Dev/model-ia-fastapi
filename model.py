from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from database import Base

class CarDetection(Base):
    __tablename__ = "car_detections"

    id = Column(Integer, primary_key=True, index=True)
    car_id = Column(Integer, index=True, nullable=False)
    car_detection_score = Column(Float, nullable=False)
    car_class = Column(String(50), nullable=True)
    car_speed = Column(Float, nullable=True)
    license_plate = Column(String(50), nullable=True)
    license_plate_score = Column(Float, nullable=True)
    bbox_x = Column(Float, nullable=False)
    bbox_y = Column(Float, nullable=False)
    bbox_w = Column(Float, nullable=False)
    bbox_h = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    def __repr__(self):
        return f"<CarDetection(id={self.id}, car_id={self.car_id}, license_plate={self.license_plate})>"
