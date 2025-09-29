

#Ajout WebSocket pour communication temps réel avec le client (index.html)
from fastapi import FastAPI
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from util import get_car, read_license_plate
from sort.sort import Sort
import base64
import json
import cv2
import numpy as np
from ultralytics import YOLO

coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')


app = FastAPI()

# Endpoint WebSocket pour recevoir les frames et renvoyer les résultats
mot_tracker = Sort()
vehicles = [2, 3, 5, 7]

@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
	await websocket.accept()
	try:
		while True:
			data = await websocket.receive_text()
			msg = json.loads(data)
			img_b64 = msg.get('video')
			if img_b64:
				img_bytes = base64.b64decode(img_b64)
				nparr = np.frombuffer(img_bytes, np.uint8)
				frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
				results = []
				# Détection véhicules
				detections = coco_model(frame)[0]
				detections_ = []
				for detection in detections.boxes.data.tolist():
					x1, y1, x2, y2, score, class_id = detection
					if int(class_id) in vehicles and score >= 0.5:
						detections_.append([x1, y1, x2, y2, score])
				track_ids = mot_tracker.update(np.asarray(detections_))
				# Détection plaques
				license_plates = license_plate_detector(frame)[0]
				for license_plate in license_plates.boxes.data.tolist():
					x1, y1, x2, y2, score, class_id = license_plate
					xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
					if car_id != -1:
						license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
						license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
						_, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
						license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
						results.append({
							'car_id': int(car_id),
							'car_bbox': [int(xcar1), int(ycar1), int(xcar2), int(ycar2)],
							'car_score': float(score),
							'license_plate_bbox': [int(x1), int(y1), int(x2), int(y2)],
							'license_plate_score': float(score),
							'license_plate_text': license_plate_text,
							'license_plate_text_score': float(license_plate_text_score)
						})
				# Ré-encode la frame pour affichage côté client
				_, buffer = cv2.imencode(".jpg", frame)
				frame_b64 = base64.b64encode(buffer).decode("utf-8")
				payload = {
					"video": frame_b64,
					"detections": [
						{
							"class": "car",
							"confidence": r['car_score'],
							"color": "N/A",
							"plates": [{
								"plate_number": r['license_plate_text'],
								"score": r['license_plate_text_score']
							}],
							"timestamp": msg.get('timestamp', '')
						} for r in results
					]
				}
				await websocket.send_json(payload)
	except WebSocketDisconnect:
		print('WebSocket déconnecté')






import asyncio
from fastapi import FastAPI, WebSocket
import numpy as np
from ultralytics import YOLO
import cv2
import time
from util import get_car, read_license_plate, draw_border
from sort.sort import Sort
import base64



def init_models():
    coco_model = YOLO('yolov8n.pt')
    license_plate_detector = YOLO('license_plate_detector.pt')
    return coco_model, license_plate_detector


def process_frame(frame, coco_model, license_plate_detector, vehicles, mot_tracker):
    results = []
    detections = coco_model(frame)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            draw_border(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 25, line_length_x=200, line_length_y=200)
            detections_.append([x1, y1, x2, y2, score])
    track_ids = mot_tracker.update(np.asarray(detections_))

    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
        if car_id != -1:
            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
            if license_plate_text is not None:
                results.append({
                    'car_id': int(car_id),
                    'car_bbox': [int(xcar1), int(ycar1), int(xcar2), int(ycar2)],
                    'license_plate_bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'license_plate_bbox_score': float(score),
                    'license_number': license_plate_text,
                    'license_number_score': float(license_plate_text_score)
                })
    return frame, results


async def generate_detections(video_path="sample.mp4"):
    coco_model, license_plate_detector = init_models()
    mot_tracker = Sort()
    vehicles = [2, 3, 5, 7]
    cap = cv2.VideoCapture(video_path)

    frame_nmr = 0
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_nmr += 1
        frame, detections = process_frame(frame, coco_model, license_plate_detector, vehicles, mot_tracker)

        # calcul FPS
        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        _, buffer = cv2.imencode(".jpg", frame)
        frameb64 = base64.b64encode(buffer).decode("utf-8")

        yield {
            "frame_nmr": frame_nmr,
            "video": frameb64,
            "fps": fps,
            "detections": detections
        }

        await asyncio.sleep(0.03)  # limiter à ~30 FPS

    cap.release()


app = FastAPI()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    esp_url = "./sample.mp4"
    async for result in generate_detections(esp_url):
        await websocket.send_json(result)
