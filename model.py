from sqlalchemy import Column, Integer, LargeBinary, String, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.sql import func
from database import Base
from sqlalchemy.orm import relationship

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
    offense = Column(Boolean, default=False, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    image_capture = Column(LargeBinary, nullable=True)
    vehicle_color = Column(String(30), nullable=False)
    
    notifications = relationship(
        "Notification",
        back_populates="car_detection",
        cascade="all, delete",           # suppression en cascade via ORM
        passive_deletes=True             # utile avec ondelete="CASCADE"
    )

    def __repr__(self):
        return f"<CarDetection(id={self.id}, car_id={self.car_id}, license_plate={self.license_plate})>"
    
class Notification(Base):
    __tablename__="notification"
    id = Column(Integer, primary_key=True, index=True)
    car_id = Column(Integer, ForeignKey("car_detections.car_id", ondelete="CASCADE"), index=True, nullable=False)
    title = Column(String(100))
    is_read = Column(Boolean, default=False)
    event_time = Column(DateTime(timezone=True), nullable=False)

    car_detection = relationship("CarDetection", back_populates="notifications")