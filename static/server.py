# @app.websocket("/ws")
# async def websocket_video(websocket: WebSocket, db: Session = Depends(get_db)):
#     await websocket.accept()
#     async for result in generate_detections("sample.mp4"):
#         for det in result["detections"]:
#             save_detection(db, det["car_id"], det["car_detection_score"], det["license_number"], det["license_number_score"], det["car_bbox"], det["vehicle_class"], det["speed_kmh"])
#         await websocket.send_json(result)
# @app.websocket("/ws/notifications")
# async def websocket_notifications(websocket: WebSocket):
#     """Canal WebSocket pour envoyer les notifications en direct"""
#     await websocket.accept()
#     connected_clients.append(websocket)
#     try:
#         while True:
#             await asyncio.sleep(1)
#     except:
#         connected_clients.remove(websocket)


# @app.on_event("startup")
# async def start_monitor():
#     asyncio.create_task(monitor_offenses(connected_clients))