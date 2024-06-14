import cv2
import numpy as np
from sort.sort import Sort
import time
import threading

# Load YOLO with GPU support
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Debug: print the result of getUnconnectedOutLayers
unconnected_out_layers = net.getUnconnectedOutLayers()
print(f"Unconnected Out Layers: {unconnected_out_layers}")

# Adjusting for different OpenCV versions
if isinstance(unconnected_out_layers, np.ndarray):
    unconnected_out_layers = unconnected_out_layers.flatten()

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

# Load COCO names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set parameters
conf_threshold = 0.5
nms_threshold = 0.4

# Initialize video
cap = cv2.VideoCapture(0)

# Initialize Sort tracker
tracker = Sort()

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()

# Frame buffer for multi-threading
frame_buffer = None
frame_lock = threading.Lock()

def process_frame():
    global frame_buffer

    while True:
        with frame_lock:
            if frame_buffer is None:
                continue
            frame = frame_buffer.copy()
            frame_buffer = None

        height, width, channels = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold and class_id == 0:  # class_id 0 is for 'person'
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        detections = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                detections.append([x, y, x + w, y + h, confidences[i]])

        # Update tracker
        tracked_objects = tracker.update(np.array(detections))

        # Draw bounding boxes and IDs
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = obj
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {int(obj_id)}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate and display FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {round(fps, 2)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Start the frame processing thread
threading.Thread(target=process_frame, daemon=True).start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    with frame_lock:
        frame_buffer = frame.copy()

cap.release()
cv2.destroyAllWindows()
