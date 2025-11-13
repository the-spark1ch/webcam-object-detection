import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть веб-камеру.")
    exit()

while True:
    success, frame = cap.read()

    if success:
        results = model(frame, stream=True, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])
                class_name = model.names[cls]

                label = f'{class_name} {conf}'

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)

                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Webcam Object Detection (Press \'q\' to quit)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()