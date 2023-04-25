import cv2
import numpy as np

# 物体検出のためのモデルとクラス名を読み込む
model = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
classes = []

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# カメラの初期化
cap = cv2.VideoCapture(0)

while True:
    # フレームの取得
    ret, frame = cap.read()

    # 物体検出の実行
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    model.setInput(blob)
    outputs = model.forward(model.getUnconnectedOutLayersNames())

    # 結果の解析と表示
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width/2)
                top = int(center_y - height/2)
                boxes.append([left, top, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        class_id = class_ids[i]
        label = f"{classes[class_id]}: {confidences[i]:.2f}"
        cv2.rectangle(frame, (left, top), (left+width, top+height), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 結果の表示
    cv2.imshow("frame", frame)
    
    # 終了するためのキー入力を待つ
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後始末
cap.release()
cv2.destroyAllWindows()
