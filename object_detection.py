import cv2
import numpy as np

# --- CRITICAL CONFIGURATION: Download these files first! ---
# You MUST download yolov3.weights and yolov3.cfg and place them in this folder.
WEIGHTS_PATH = "yolov3.weights"
CONFIG_PATH = "yolov3.cfg"
# --------------------------------------------------

# 1. Load the Network (YOLO)
try:
    net = cv2.dnn.readNet(WEIGHTS_PATH, CONFIG_PATH)
except:
    # Print error and exit if files are missing
    print("Error: Could not load YOLO files. Please ensure 'yolov3.weights' and 'yolov3.cfg' are in the same folder.")
    exit()

# 2. Get output layer names
layer_names = net.getLayerNames()
# Find the names of the unconnected output layers, which are the output layers of the YOLO model
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# 3. Start Video Capture (Webcam)
cap = cv2.VideoCapture(0)  # 0 means your default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # 4. Preprocessing and Detection
    # Create a 4D blob from image, scale pixel values, resize to 416x416
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # 5. Drawing Boxes and Labels
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # We only care about detections with high confidence (e.g., > 50%)
            if confidence > 0.5:
                # Get box coordinates (center_x, center_y, width, height)
                x, y, w, h = (detection[0:4] * [width, height, width, height]).astype("int")
                
                # Draw box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Display confidence score as label
                cv2.putText(frame, f"Object: {round(confidence*100)}%", (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # 6. Display the result
    cv2.imshow("Object Detection - CodeAlpha", frame)

    # Quit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
