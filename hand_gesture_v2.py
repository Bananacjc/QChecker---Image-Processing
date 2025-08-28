import cv2
import mediapipe as mp
import numpy as np
import time
from abc import ABC, abstractmethod

class HandGesture(ABC):
    
    @abstractmethod
    def detect(self, landmarks: list) -> bool:
        '''
        Return True if this gesture is detected from landmarks
        '''
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass

class OKSignGesture(HandGesture):
    def __init__(self, threshold=50):
        self.threshold = threshold
    
    @property
    def name(self):
        return "OK Sign"
    
    def detect(self, landmarks):
        thumb_tip = np.array(landmarks[4])
        index_tip = np.array(landmarks[8])
        distance = np.linalg.norm(thumb_tip - index_tip)
        
        middle_extended = landmarks[12][1] < landmarks[10][1]
        ring_extended = landmarks[16][1] < landmarks[14][1]
        pinky_extended = landmarks[20][1] < landmarks[18][1]
        
        return distance < self.threshold and middle_extended and ring_extended and pinky_extended

class RudeGesture(HandGesture):
    def __init__(self):
        pass

    @property
    def name(self):
        return "Rude Gesture"

    def detect(self, landmarks):
        middle_up = landmarks[12][1] < landmarks[10][1]
        index_down = landmarks[8][1] > landmarks[6][1]
        ring_down = landmarks[16][1] > landmarks[14][1]
        pinky_down = landmarks[20][1] > landmarks[18][1]

        return middle_up and index_down and ring_down and pinky_down

class ThumbsUpGesture(HandGesture):
    def __init__(self, extended_thres=0.7, vertical_thres = 20):
        self.extended_thres = extended_thres
        self.vertical_thres = vertical_thres
    
    @property
    def name(self):
        return "Thumbs Up"
    
    def detect(self, landmarks):
        import math
        
        def euclidean(a, b):
            return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

        thumb_tip = landmarks[4]
        index_mcp = landmarks[5]
        thumb_extended = euclidean(thumb_tip, index_mcp) > self.extended_thres
        
        dx = landmarks[4][0] - landmarks[0][0]
        dy = landmarks[4][1] - landmarks[0][1]
        thumb_angle = math.degrees(math.atan2(dy, dx))
        thumb_is_vertical = abs(abs(thumb_angle) - 90) < self.vertical_thres
        
        fingers_folded = (
            landmarks[8][1] > landmarks[5][1] and
            landmarks[12][1] > landmarks[9][1] and
            landmarks[16][1] > landmarks[13][1] and
            landmarks[20][1] > landmarks[17][1]
        )

    
        return thumb_extended and thumb_is_vertical and fingers_folded

class HandGestureRecognizer:
    def __init__(self, model_path="hand_landmarker.task", 
                 detection_conf=0.3, presence_conf=0.3, tracking_conf=0.3,
                 max_hands=1):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.last_result = None
        self.gestures: list[HandGesture] = []  # registered gestures

        def _callback(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int): # type: ignore
            self.last_result = result

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=_callback,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=presence_conf,
            min_tracking_confidence=tracking_conf,
            num_hands=max_hands
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def add_gesture(self, gesture: HandGesture):
        self.gestures.append(gesture)
    
    def draw(self, landmarks, frame):
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
        connection_pairs = [
                (0,1),(1,2),(2,3),(3,4),            # thumb
                (5,6),(6,7),(7,8),                  # index
                (9,10),(10,11),(11,12),             # middle
                (13,14),(14,15),(15,16),            # ring
                (17,18),(18,19),(19,20),            # pinky
                (0,5),(5,9),(9,13),(13,17),(0,17)   # palm connections
        ]
            
        for start, end in connection_pairs:
            if start < len(landmarks) and end < len(landmarks):
                cv2.line(frame, landmarks[start], landmarks[end], (0, 255, 255), 2)
    
    def detect_gesture(self, landmarks):
        for gesture in self.gestures:
            if gesture.detect(landmarks):
                return gesture.name
        return None

                
    def run(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            self.landmarker.detect_async(mp_image, timestamp)

            if self.last_result and self.last_result.hand_landmarks:
                detected_gesture = None
                for hand_landmarks in self.last_result.hand_landmarks:
                    landmarks = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) 
                                for lm in hand_landmarks]

                    self.draw(landmarks, frame)

                    if detected_gesture is None:
                        detected_gesture = self.detect_gesture(landmarks)

                if detected_gesture:
                    cv2.putText(frame, f"Gesture: {detected_gesture}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Hand Gesture Recognizer', frame)
            
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    system = HandGestureRecognizer(max_hands=2)
    
    system.add_gesture(OKSignGesture(threshold=23))
    system.add_gesture(RudeGesture())
    system.add_gesture(ThumbsUpGesture())
    system.run(camera_id=0)