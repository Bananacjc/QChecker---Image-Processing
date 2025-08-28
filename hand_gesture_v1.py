import cv2
import mediapipe as mp
import numpy as np
import time
import webbrowser

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

last_result = None

def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global last_result
    last_result = result

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

cap = cv2.VideoCapture(0)
middle_finger_detected = False
middle_start_time = None
opened_youtube = False
frame_count = 0

with GestureRecognizer.create_from_options(options) as recognizer:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frame_count += 1
        frame = cv2.resize(frame, (640, 480))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        recognizer.recognize_async(mp_image, timestamp)
        
        if last_result and last_result.hand_landmarks:
            for hand_landmarks in last_result.hand_landmarks:
                landmarks = []
                for idx, landmark in enumerate(hand_landmarks):
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    landmarks.append((x, y))
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(frame, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

                connection_pairs = [(0,1), (1,2), (2,3), (3,4), (5,6), (6,7), (7,8),
                                    (9,10), (10,11), (11,12), (13,14), (14,15), (15,16),
                                    (17,18), (18,19), (19,20)]
                for start, end in connection_pairs:
                    if start < len(landmarks) and end < len(landmarks):
                        cv2.line(frame, landmarks[start], landmarks[end], (0, 255, 255), 2)
                
                # ---- CUSTOM OK SIGN DETECTION ----
                thumb_tip = np.array(landmarks[4])
                index_tip = np.array(landmarks[8])
                distance = np.linalg.norm(thumb_tip - index_tip)
                middle_extended = landmarks[12][1] < landmarks[10][1]
                ring_extended   = landmarks[16][1] < landmarks[14][1]
                pinky_extended  = landmarks[20][1] < landmarks[18][1]
                
                # ---- WORLDWIDE LANGUAGE DETECTION ----
                middle_up = landmarks[12][1] < landmarks[10][1]
                index_down = landmarks[8][1] > landmarks[6][1]
                ring_down = landmarks[16][1] > landmarks[14][1]
                pinky_down = landmarks[20][1] > landmarks[18][1]
                thumb_down = landmarks[4][1] > landmarks[2][1]
                
                if middle_up and index_down and ring_down and pinky_down:
                    cv2.putText(frame, "Custom Gesture: That's rude", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    if middle_start_time is None:
                        middle_start_time = time.time()
                    elif time.time() - middle_start_time >= 2:
                        middle_finger_detected = True
                        if not(opened_youtube):
                            webbrowser.open("https://youtu.be/yFE6qQ3ySXE")
                            opened_youtube = True
                        break
                        
                else:
                    middle_start_time = None

                # Heuristic threshold (adjust for your camera resolution)
                if distance < 40 and middle_extended and ring_extended and pinky_extended:  
                    cv2.putText(frame, "Custom Gesture: OK Sign", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        if last_result and last_result.gestures:
            gesture = last_result.gestures[0][0].category_name
            score = last_result.gestures[0][0].score
            cv2.putText(frame, f'Gesture: {gesture} ({score:.2f})', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Gesture Control', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        if middle_finger_detected:
            break

cap.release()
cv2.destroyAllWindows()

