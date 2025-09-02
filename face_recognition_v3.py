import os
import cv2
import json
import time
import math
import numpy as np
import statistics
import mediapipe as mp
from abc import ABC, abstractmethod
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
import threading
import queue
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import queue as pyqueue

# HAND GESTURE
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

# FACE RECOGNITION
def get_db():
    # Load variables from the .env file into the environment
    load_dotenv(dotenv_path='./.env')

    # Get the FIREBASE_CREDENTIALS variable
    raw = os.environ.get("FIREBASE_CREDENTIALS")
    if not raw:
        raise RuntimeError("Missing FIREBASE_CREDENTIALS in your .env file")

    # Parse JSON and fix private key newlines
    cfg = json.loads(raw)
    cfg["private_key"] = cfg["private_key"].replace("\n", "\n")

    if not firebase_admin._apps:
        cred = credentials.Certificate(cfg)
        firebase_admin.initialize_app(cred)

    return firestore.client()

def fetch_students(only_registered=None):
    """
    Read Firestore collection 'students' and return a list of dicts:
    seat_num (str), student_id (str), name, course, award, image_path
    Set only_registered=True/False to filter; None returns all.
    """
    db = get_db()
    col = db.collection("students")
    q = col

    if only_registered is not None:
        q = q.where("registered", "==", bool(only_registered))

    # sort by seat number if you want deterministic order
    q = q.order_by("seat_num")

    result = []
    for snap in q.stream():
        data = snap.to_dict() or {}
        result.append({
            "seat_num": str(data.get("seat_num", "")),
            "student_id": snap.id,
            "name": data.get("name", ""),
            "course": data.get("course", ""),
            "award": data.get("award", ""),
            "image_path": data.get("image_path", ""),
        })
    return result

def fetch_student_by_id(student_id: str):
    db = get_db()
    doc = db.collection("students").document(student_id).get()
    return doc.to_dict() if doc.exists else None

class ImageAugmenter:
    def __init__(self, 
                 brightness_shifts:list[int] | list[float]=[-40, -20, 20, 40], 
                 contrast_scales:list[int] | list[float]=[0.8, 1.2], 
                 rotation_angles:list[int] | list[float]=[-15, 15]):
        '''
        Parameters
        ----------
        brightness_shifts: list[int, float]
            Shift pixels intensity
        contrast_scales: list[int, float]
            Multipliers for constrast adjustment
        rotation_angles: list[int, float]
            Angles in degress for rotation
        '''
    
        self.brightness_shifts = brightness_shifts
        self.contrast_scales = contrast_scales
        self.rotation_angles = rotation_angles
    


    def augment(self, img: cv2.typing.MatLike):
        '''
        Produce image variation 
        '''
        augmented = [img]  # Original

        # Brightness
        for beta in self.brightness_shifts:
            bright = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
            augmented.append(bright)

        # Contrast
        for alpha in self.contrast_scales:
            contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
            augmented.append(contrast)

        # Rotations
        h, w = img.shape[:2]
        for angle in self.rotation_angles:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            rotated = cv2.warpAffine(img, M, (w, h))
            augmented.append(rotated)

        return augmented

class Detector(ABC):
    @property
    def name(self) -> str:
        pass

    
    @abstractmethod
    def detect_faces(self, frame):
        pass
    
class SkinSegmentationDetector(Detector):
        def __init__(self, color_space="ycrcb", 
                     ksize = 5, 
                     ycrcb_min=[0, 133, 77], 
                     ycrcb_max=[255, 173, 127],
                     hsv_min=[0, 40, 60],
                     hsv_max=[50, 150, 255],
                     width_min=30,
                     height_min=30,
                     aspect_ratio_min=1.0,
                     aspect_ratio_max=1.9,
                     frame_area_min=0.01,
                     frame_area_max=0.7,
                     debug=False):
            
            self.color_space = color_space.lower()
            self.ycrcb_min = np.array(ycrcb_min, np.uint8)
            self.ycrcb_max = np.array(ycrcb_max, np.uint8)
            self.hsv_min = np.array(hsv_min, np.uint8)
            self.hsv_max = np.array(hsv_max, np.uint8)
            
            self.ksize = ksize
            
            self.width_min = width_min
            self.height_min = height_min
            self.aspect_ratio_min = aspect_ratio_min
            self.aspect_ratio_max = aspect_ratio_max
            self.frame_area_min = frame_area_min
            self.frame_area_max = frame_area_max
            
            self.debug = debug
            
        @property
        def name(self):
            return "Skin Segmentation Face Detector"
        
        def segment(self, frame):
            '''
            Perform skin segmentation on a single frame
            Returns a binary mask
            '''
            
            if self.color_space == 'ycrcb':
                converted = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
                mask = cv2.inRange(converted, self.ycrcb_min, self.ycrcb_max)
            elif self.color_space == 'hsv':
                converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(converted, self.hsv_min, self.hsv_max)
            else:
                raise ValueError("Unsupported color space, please only choose 'ycrcb' or 'hsv'")
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.ksize, self.ksize))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            return mask
        
        def detect_faces(self, frame):
            '''
            Detect face regions based on skin segmentation
            Returns bounding boxes (x, y, w, h)
            '''
            
            mask = self.segment(frame)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            faces = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = h / float(w)
                area = w*h
                frame_area = frame.shape[0] * frame.shape[1]
                
                if (w > self.width_min and h > self.height_min and 
                    self.aspect_ratio_min <= aspect_ratio <= self.aspect_ratio_max and
                    self.frame_area_min*frame_area < area < self.frame_area_max*frame_area):
                    faces.append((x,y,w,h))
            
            if not faces:
                faces = None
            
            return faces, (mask if self.debug else None)

class ViolaJonesDetector(Detector):
        def __init__(self, cascade_path=None, 
                     scaleFactor=1.1, 
                     minNeighbors=3, 
                     minSize=(50, 50),
                     debug=False):
            if cascade_path is None:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
            else:
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                
            self.scaleFactor = scaleFactor
            self.minNeighbors = minNeighbors
            self.minSize = minSize
            
            self.debug = debug
        
        @property
        def name(self):
            return "Viola-Jones Face Detector"
        
        
        def detect_faces(self, frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scaleFactor,
                minNeighbors=self.minNeighbors,
                minSize=self.minSize
            )
            
            return faces, (gray if self.debug else None)

# class HOGDetector(Detector):
#     def __init__(self, debug=False):
#         import dlib
#         self.hog_detector = dlib.get_frontal_face_detector()
#         self.debug = debug
    
#     @property
#     def name(self):
#         return "HOG Face Detector"
    
#     def detect_faces(self, frame):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = self.hog_detector(gray, 1)
#         boxes = [(d.left(), d.top(), d.width(), d.height()) for d in faces]
#         return boxes, (gray if self.debug else None)

class DNNDetector(Detector):
    def __init__(self, prototxt, model, conf_threshold=0.5, debug=False):
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.conf_threshold = conf_threshold
        self.debug = debug
    
    @property
    def name(self):
        return "DNN Face Detector"
    
    def detect_faces(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                boxes.append((x1, y1, x2-x1, y2-y1))
        return boxes, (frame if self.debug else None)

class FacePreprocessor:
    def __init__(self, size=(100, 100), normalize=True, equalize=True, crop_tight=True, debug=False):
        self.size = size
        self.normalize = normalize
        self.equalize = equalize
        self.crop_tight = crop_tight
        self.debug = debug
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    
    def _align_face(self, gray_face):
        eyes = self.eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) >= 2:
            # Pick the two largest eyes
            eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]
            eye_centers = [(x + w//2, y + h//2) for (x, y, w, h) in eyes]

            # Ensure left & right order
            eye_centers = sorted(eye_centers, key=lambda p: p[0])
            left_eye, right_eye = eye_centers

            # Compute angle
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))

            # Compute scale (normalize eye distance to a fixed size)
            dist = np.sqrt(dx**2 + dy**2)
            desired_dist = self.size[0] * 0.4  # 40% of width
            scale = desired_dist / dist

            # Compute rotation matrix
            eyes_center = ((left_eye[0] + right_eye[0]) // 2.0,
                           (left_eye[1] + right_eye[1]) // 2.0)
            M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

            # Apply affine transform
            aligned = cv2.warpAffine(gray_face, M, self.size, flags=cv2.INTER_CUBIC)
            return aligned

        return None  # if no eyes found
        
    def preprocess(self, frame, face_box):
        x, y, w, h = face_box
        
        face = frame[y:y+h, x:x+w]
        
        if self.crop_tight:
            mx = int(0.1 * w)
            my = int(0.1 * h)
            x1 = max(0, mx)
            y1 = max(0, my)
            x2 = w - mx
            y2 = h - my
            face = face[y1:y2, x1:x2]
        
        face = cv2.resize(face, self.size)
        
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        aligned = self._align_face(face_gray)
        if aligned is not None:
            face_gray = aligned
        
        if self.equalize:
            face_gray = cv2.equalizeHist(face_gray)
        
        if self.normalize:
            face_gray = face_gray.astype("float32") / 255.0
            
        return face_gray

class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, face_img: np.ndarray) -> np.ndarray:
        '''
        Convert a normalized face into a feature vecto
        '''
        pass

class PCAFeatureExtractor(FeatureExtractor):
    '''
    Eigenfaces, Simplest, Sensitive to lightning & backgrounf\n
    For complex data, use num_components=80-150\n
    '''
    def __init__(self, num_components=50):
        self.num_components = num_components
        self.mean = None
        self.components = None
        
    def fit(self, faces: list[np.ndarray]):
        '''
        faces: list of images
        '''
        X = np.array([f.flatten() for f in faces])
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.components = Vt[:self.num_components]
    
    def extract(self, face_img: np.ndarray) -> np.ndarray:
        x = face_img.flatten()
        X_centered = x - self.mean
        return np.dot(self.components, X_centered)

class LDAFeatureExtractor(FeatureExtractor):
    '''
    Fisherfaces\n
    More data is better\n
    Can be combined with PCA to reduce noise
    pca_components too small = loses info; Too big = overfits
    '''
    def __init__(self, num_components=None):
        self.num_components = num_components
        self.projection_matrix = None
        self.mean = None
    
    def fit(self, faces: list[np.ndarray], labels: list[int]):
        X = np.array([f.flatten() for f in faces])
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        classes = np.unique(labels)
        
        Sw = np.zeros((X.shape[1], X.shape[1]))
        Sb = np.zeros((X.shape[1], X.shape[1]))
        
        for c in classes:
            Xc = X_centered[np.array(labels) == c]
            mean_c = np.mean(Xc, axis=0)
            Sw += (Xc - mean_c).T @ (Xc - mean_c)
            n_c = Xc.shape[0]
            mean_diff = (mean_c - self.mean).reshape(-1,1)
            Sb += n_c * (mean_diff @ mean_diff.T)
        
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
        idx = np.argsort(-eigvals.real)
        eigvecs = eigvecs[:, idx]
        
        if self.num_components is None:
            self.num_components = len(classes) - 1
        
        self.projection_matrix = eigvecs[:, :self.num_components].real
        
    def extract(self, face_img: np.ndarray) -> np.ndarray:
        x = face_img.flatten()
        x_centered = x - self.mean
        return np.dot(x_centered, self.projection_matrix)

class FastLDAFeatureExtractor(FeatureExtractor):
    def __init__(self, num_components=None, pca_components=50):
        """
        num_components: LDA components (<= num_classes - 1)
        pca_components: number of PCA components to reduce dimensionality
        """
        self.num_components = num_components
        self.pca_components = pca_components
        self.mean = None
        self.pca_components_matrix = None
        self.lda_projection = None

    def fit(self, faces: list[np.ndarray], labels: list[int]):
        # Flatten faces
        X = np.array([f.flatten() for f in faces])
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        labels = np.array(labels)
        classes = np.unique(labels)

        # --- Step 1: PCA ---
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        self.pca_components_matrix = Vt[:self.pca_components].T  # [D, pca_components]
        X_pca = X_centered @ self.pca_components_matrix          # [n_samples, pca_components]

        # --- Step 2: LDA in PCA space ---
        dim = X_pca.shape[1]
        Sw = np.zeros((dim, dim))
        Sb = np.zeros((dim, dim))
        overall_mean = np.mean(X_pca, axis=0)

        for c in classes:
            Xc = X_pca[labels == c]
            mean_c = np.mean(Xc, axis=0)
            Sw += (Xc - mean_c).T @ (Xc - mean_c)
            n_c = Xc.shape[0]
            mean_diff = (mean_c - overall_mean).reshape(-1, 1)
            Sb += n_c * (mean_diff @ mean_diff.T)

        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
        idx = np.argsort(-eigvals.real)
        eigvecs = eigvecs[:, idx]

        if self.num_components is None:
            self.num_components = len(classes) - 1

        self.lda_projection = eigvecs[:, :self.num_components].real  # [pca_components, lda_components]

    def extract(self, face_img: np.ndarray) -> np.ndarray:
        x = face_img.flatten()
        x_centered = x - self.mean
        x_pca = x_centered @ self.pca_components_matrix
        x_lda = x_pca @ self.lda_projection
        return x_lda

class LBPFeatureExtractor:
    '''
    Robust to lightning, sentive to alignment and scale\n
    small grid_x & grid_y capture more details
    '''
    def __init__(self, grid_x=16, grid_y=16):
        self.grid_x = grid_x
        self.grid_y = grid_y
    
    def _lbp(self, img):
        lbp_img = np.zeros_like(img, dtype=np.uint8)
        for y in range(1, img.shape[0]-1):
            for x in range(1, img.shape[1]-1):
                center = img[y, x]
                code = 0
                code |= (img[y-1, x-1] > center) << 7
                code |= (img[y-1, x]   > center) << 6
                code |= (img[y-1, x+1] > center) << 5
                code |= (img[y, x+1]   > center) << 4
                code |= (img[y+1, x+1] > center) << 3
                code |= (img[y+1, x]   > center) << 2
                code |= (img[y+1, x-1] > center) << 1
                code |= (img[y, x-1]   > center) << 0
                lbp_img[y, x] = code
        return lbp_img
    
    def extract(self, face_img: np.ndarray) -> np.ndarray:
        h, w = face_img.shape
        lbp_img = self._lbp((face_img * 255).astype(np.uint8))
        grid_h, grid_w = h // self.grid_y, w // self.grid_x
        features = []
        for i in range(self.grid_y):
            for j in range(self.grid_x):
                patch = lbp_img[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                hist, _ = np.histogram(patch, bins=256, range=(0, 256))
                hist = hist.astype("float32") / (hist.sum() + 1e-6)
                features.extend(hist)
        
        features = np.array(features)
        features /= np.linalg.norm(features) + 1e-6
        return features

class FaceRecognizer:
    def __init__(self, extractor: 'FeatureExtractor' = None):
        self.extractor = extractor if extractor else PCAFeatureExtractor()
        self.database = []  # list of (feature_vector, label)
        self._trained = False
        self.threshold = None
        self.distance_func = None

    def fit(self, faces, labels=None):
        # Fit PCA or LDA depending on extractor type
        if isinstance(self.extractor, PCAFeatureExtractor):
            self.extractor.fit(faces)
            self.distance_func = self._chi2_distance
        elif isinstance(self.extractor, LDAFeatureExtractor) or isinstance(self.extractor, FastLDAFeatureExtractor):
            self.extractor.fit(faces, labels)
            self.distance_func = self._chi2_distance
        elif isinstance(self.extractor, LBPFeatureExtractor):
            self.distance_func = self._chi2_distance
        else:
            raise ValueError(f"Unknown extractor type: {type(self.extractor)}")

        self._trained = True

    def enroll(self, face_img, label):
        if not self._trained:
            raise RuntimeError("Extractor not trained. Call fit() before enrollment.")
        feat = self.extractor.extract(face_img)
        self.database.append((feat, label))
        self._update_threshold()

    def _update_threshold(self):
        """Compute a dynamic threshold based on intra-class distances."""
        if not self.database:
            self.threshold = None
            return
        
        intra_dists = []
        labels = [label for _, label in self.database]
        feats = [feat for feat, _ in self.database]

        # Compute distances between all same-label pairs
        for i in range(len(feats)):
            for j in range(i + 1, len(feats)):
                if labels[i] == labels[j]:
                    d = self.distance_func(feats[i], feats[j])
                    intra_dists.append(d)
        
        if intra_dists:
            self.threshold = max(intra_dists) * 1.3 # allow small margin
        else:
            self.threshold = 0.3  # fallback

    def recognize(self, face_img):
        
        if not self._trained or not self.database:
            return "Unknown"

        feat = self.extractor.extract(face_img)
        min_dist, best_label = float("inf"), None

        for db_feat, label in self.database:
            dist = self.distance_func(feat, db_feat)
            if dist < min_dist:
                min_dist, best_label = dist, label

        if self.threshold is None:
            self._update_threshold()

        return best_label if min_dist < self.threshold else "Unknown"
    
    def recognize_knn(self, face_img, k=3):
        """Recognize a face using nearest neighbor (k=1) or kNN classification."""
        if not self._trained or not self.database:
            return "Unknown"

        feat = self.extractor.extract(face_img)

        # Compute all distances
        distances = [(self.distance_func(feat, db_feat), label) 
                     for db_feat, label in self.database]
        distances.sort(key=lambda x: x[0])

        if k == 1:
            # --- Standard nearest neighbor ---
            min_dist, best_label = distances[0]
            if self.threshold is None:
                self._update_threshold()
            return best_label if min_dist < self.threshold else "Unknown"
        else:
            # --- kNN majority vote ---
            top_k = distances[:k]
            labels = [label for _, label in top_k]

            # majority voting
            best_label = max(set(labels), key=labels.count)

            # check threshold with the closest neighbor
            if self.threshold is None:
                self._update_threshold()
            return best_label if top_k[0][0] < self.threshold else "Unknown"
        

    # --- Distance functions ---
    @staticmethod
    def _euclidean(a, b):
        return np.linalg.norm(a - b)

    @staticmethod
    def _chi2_distance(a, b, eps=1e-10):
        return 0.5 * np.sum(((a - b) ** 2) / (a + b + eps))
    
class FacePipeline:
    def __init__(self, students, debug=False, augmenter= None, detector=None, extractor=None):
        self.preprocessor = FacePreprocessor(debug=debug)
        self.augmenter = augmenter or ImageAugmenter()
        self.detector = detector or ViolaJonesDetector()
        self.recognizer = FaceRecognizer(extractor or PCAFeatureExtractor())
        self.students = students
        self.detector.debug = debug

        self._load_datasets()

    def _load_datasets(self):
        training_faces = []
        training_labels = []

        for student in self.students:
            img = cv2.imread(f"./known_faces/{student['image_path']}")
            if img is None:
                print(f"Could not load {student['image_path']}")
                continue
            
            faces, _ = self.detector.detect_faces(img)
            if faces is None:
                print(f"No faces in {student['image_path']}")
                continue
            
            x, y, w, h = faces[0]
            face = self.preprocessor.preprocess(img, (x, y, w, h))
                
            augmented_faces = self.augmenter.augment(face)
                
            for aug in augmented_faces:
                training_faces.append(aug)
                training_labels.append(student['name'])

        if isinstance(self.recognizer.extractor, LDAFeatureExtractor) or isinstance(self.recognizer.extractor, FastLDAFeatureExtractor):
            self.recognizer.fit(training_faces, training_labels)
        elif isinstance(self.recognizer.extractor, PCAFeatureExtractor):
            self.recognizer.fit(training_faces)
        else:
            self.recognizer.fit(None)

        for face, sid in zip(training_faces, training_labels):
            self.recognizer.enroll(face, sid)

    def run(self, camera_index=0): 
        cap = cv2.VideoCapture(camera_index)
    
        while True:
            ret, frame = cap.read()
            if not ret:
                break
        
            faces, other = self.detector.detect_faces(frame)
            
            if faces is not None:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    face_img = self.preprocessor.preprocess(frame, (x, y, w, h))
                    
                    if self.preprocessor.debug:
                        cv2.imshow("Preprocessed Face", face_img)
                        
                    # Switch here
                    sid = self.recognizer.recognize_knn(face_img, k=1)
                    
                    if sid != "Unknown":
                        info = next((d for d in self.students if d["name"] == sid), None)
                        if info:
                            text = f"{info['name']} ({info['award']})"
                        else:
                            text = sid
                    else:
                        text = "Unkown"
                    
                    cv2.putText(frame, text, (x,y-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                
            cv2.imshow(self.detector.name, frame)
            
            if other is not None:
                cv2.imshow(f"{self.detector.name} Debug", other)
            
            if cv2.waitKey(5) & 0xFF in [27, ord('q')]:
                break
        
        cap.release()
        cv2.destroyAllWindows()

class FaceComparator:
    def __init__(self, detector=None, extractor=None, preprocessor=None, threshold=0.3):
        self.detector = detector or ViolaJonesDetector()
        self.preprocessor = preprocessor or FacePreprocessor()
        self.extractor = extractor or LBPFeatureExtractor()
        self.threshold = threshold

    def _get_face(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        faces, _ = self.detector.detect_faces(img)
        if faces is None or len(faces) == 0:
            raise ValueError(f"No face detected in {img_path}")
        
        x, y, w, h = faces[0]
        face = self.preprocessor.preprocess(img, (x, y, w, h))
        return face

    def compare(self, img1_path, img2_path, use_chi2=True):
        # Preprocess both
        face1 = self._get_face(img1_path)
        face2 = self._get_face(img2_path)

        # Extract features
        feat1 = self.extractor.extract(face1)
        feat2 = self.extractor.extract(face2)

        # Compute distance
        if use_chi2:
            dist = FaceRecognizer._chi2_distance(feat1, feat2)
        else:
            dist = FaceRecognizer._euclidean(feat1, feat2)

        # Decide match
        same = dist < self.threshold
        return {"distance": dist, "same_person": same}

# ===========================
# ORCHESTRATOR (ceremony flow)
# ===========================
from collections import deque

CAM_INDEX = 0
OK_REQUIRED = True
MATCH_TIMEOUT_SEC = 10.0
# --- Two-stage compare (LBP first, PCA fallback) ---
LBP_DISTANCE_THRESHOLD = 0.10      # your current tuned value
PCA_DISTANCE_THRESHOLD = 0.35      # start here; tune on your data
PCA_COMPONENTS = 60                # 50–120 is usually fine
PCA_TIMEOUT_SEC = 8.0              # how long to try PCA after LBP fails


# ---- helpers ----
def sort_by_seat(students):
    def _to_int(s):
        try: return int(s.get("seat_num", 0))
        except: return 0
    return sorted(students, key=_to_int)

def try_decode_qr(frame_bgr):
    qr = cv2.QRCodeDetector()
    try:
        data, pts, _ = qr.detectAndDecode(frame_bgr)
    except cv2.error:
        return None
    
    if pts is not None and len(pts) > 0 and data:
        try:
            return json.loads(data)
        except Exception:
            return None
    return None

def build_feature_db(students, detector, preproc, extractor):
    per_face = {}
    ok_cnt = miss_cnt = 0
    for s in students:
        sid = s.get("student_id", "")
        img_path = os.path.join("./known_faces", s.get("image_path", ""))
        img = cv2.imread(img_path)
        if img is None:
            print(f"[train] Missing image: {img_path}")
            miss_cnt += 1
            continue
        faces, _ = detector.detect_faces(img)
        empty = (faces is None or
                 (hasattr(faces, "size") and faces.size == 0) or
                 (hasattr(faces, "__len__") and len(faces) == 0))
        if empty:
            print(f"[train] No face detected: {img_path}")
            miss_cnt += 1
            continue
        boxes = faces.tolist() if hasattr(faces, "tolist") else list(faces)
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        face = preproc.preprocess(img, (x, y, w, h))
        per_face[sid] = face
        ok_cnt += 1

    if isinstance(extractor, PCAFeatureExtractor) and per_face:
        extractor.fit(list(per_face.values()))
    elif isinstance(extractor, (LDAFeatureExtractor, FastLDAFeatureExtractor)) and per_face:
        sids = list(per_face.keys())
        lbl_map = {sid:i for i, sid in enumerate(sids)}
        faces = [per_face[sid] for sid in sids]
        labels = [lbl_map[sid] for sid in sids]
        extractor.fit(faces, labels)

    db = {sid: extractor.extract(face) for sid, face in per_face.items()}
    print(f"[train] Features ready for {ok_cnt} students; {miss_cnt} skipped.")
    return db

def detect_ok_sign(gesture_sys, frame_bgr):
    H, W = frame_bgr.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_bgr)
    ts = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    gesture_sys.landmarker.detect_async(mp_image, ts)
    if gesture_sys.last_result and hasattr(gesture_sys.last_result, "hand_landmarks") and gesture_sys.last_result.hand_landmarks:
        for hand_landmarks in gesture_sys.last_result.hand_landmarks:
            landmarks = [(int(lm.x * W), int(lm.y * H)) for lm in hand_landmarks]
            gesture_sys.draw(landmarks, frame_bgr)
            if gesture_sys.detect_gesture(landmarks) == "OK Sign":
                return True
    return False

# STAFF PANEL
class StaffPanelTk:
    """
    Tiny Tkinter panel that runs in its own thread.
    Buttons enqueue commands: 'force', 'skip', 'call', 'quit'
    update_status(...) updates the UI; get_command_nowait() polls commands.
    """
    def __init__(self, title="Staff Panel"):
        import threading, queue, tkinter as tk
        from tkinter import ttk, messagebox

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self._title = title
        self._cmd_q: "queue.Queue[str]" = queue.Queue()
        self._alive = True

        # photo support
        self.photo_label = None
        self.photo_cache = None
        self._btn_call = None

        # status snapshot for UI thread
        self._status_lock = threading.Lock()
        self._status = {
            "expected": "(none)",
            "index": 0,
            "total": 0,
            "alert": "",
            "next_hint": "",
            "queue_preview": [],
            "expected_id": None
        }

        self._thread = threading.Thread(target=self._run_tk, daemon=True)
        self._thread.start()

    # ---- public API ----
    def update_status(self, expected, idx, total, q, alert_text=None, next_hint=None):
        """Called from the OpenCV thread each frame."""
        preview = []
        for i, s in enumerate(list(q)[:5]):  # show first 5 in queue
            preview.append(f"{i+1}. Seat {s.get('seat_num','?')} — {s.get('name','?')} — {s.get('award','')}")

        exp_str = "(none)"
        exp_id = None
        if expected:
            exp_str = f"Seat {expected.get('seat_num','?')} — {expected.get('name','?')}"
            exp_id = expected.get("student_id")

        with self._status_lock:
            self._status.update({
                "expected": exp_str,
                "expected_id": exp_id,
                "index": int(idx),
                "total": int(total),
                "alert": alert_text or "",
                "next_hint": next_hint or "",
                "queue_preview": preview
            })

    def get_command_nowait(self):
        try:
            return self._cmd_q.get_nowait()
        except Exception:
            return None

    def shutdown(self):
        self._alive = False

    # Exposed helper: enable/disable the Call button (thread-safe)
    def set_call_enabled(self, enabled: bool):
        try:
            def _apply():
                if not self._btn_call: return
                if enabled:
                    self._btn_call.state(["!disabled"])
                else:
                    self._btn_call.state(["disabled"])
            self._root.after(0, _apply)
        except Exception:
            pass

    # ---- UI thread ----
    def _on_quit_clicked(self):
        if self.messagebox.askyesno("Confirm", "Quit the system?"):
            self._cmd_q.put("quit")

    def _run_tk(self):
        tk = self.tk
        ttk = self.ttk

        self._root = tk.Tk()
        self._root.title(self._title)
        self._root.attributes("-topmost", True)
        self._root.protocol("WM_DELETE_WINDOW", self._on_quit_clicked)

        # Vars
        self.var_expected = tk.StringVar(value="Expected: (none)")
        self.var_index    = tk.StringVar(value="Index: 0/0")
        self.var_next     = tk.StringVar(value="Next: -")
        self.var_alert    = tk.StringVar(value="")

        # Layout
        frm = ttk.Frame(self._root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        self._root.columnconfigure(0, weight=1)
        self._root.rowconfigure(0, weight=1)

        lbl_title = ttk.Label(frm, text="== Staff Panel ==", font=("Segoe UI", 12, "bold"))
        lbl_title.grid(row=0, column=0, columnspan=5, sticky="w", pady=(0,6))

        ttk.Label(frm, textvariable=self.var_expected).grid(row=1, column=0, columnspan=5, sticky="w")
        ttk.Label(frm, textvariable=self.var_index).grid(row=2, column=0, columnspan=5, sticky="w")
        ttk.Label(frm, textvariable=self.var_next).grid(row=3, column=0, columnspan=5, sticky="w")

        # Expected student photo
        self.photo_label = ttk.Label(frm, text="(No image)")
        self.photo_label.grid(row=4, column=0, columnspan=5, pady=(8,0))

        # Queue list
        ttk.Label(frm, text="Queue (first 5):").grid(row=5, column=0, columnspan=5, sticky="w", pady=(8,0))
        self.list_queue = tk.Listbox(frm, height=5)
        self.list_queue.grid(row=6, column=0, columnspan=5, sticky="nsew")

        # Buttons row
        btns = [
            ("Force",   "force"),
            ("Skip",    "skip"),
            ("Call next","call"),
            ("Quit",    "quit"),
        ]
        for i, (txt, cmd) in enumerate(btns):
            if cmd == "quit":
                action = self._on_quit_clicked
            else:
                action = lambda c=cmd: self._cmd_q.put(c)
            b = ttk.Button(frm, text=txt, command=action)
            b.grid(row=7, column=i, padx=2, pady=(10,0), sticky="ew")
            if cmd == "call":
                self._btn_call = b

        # Alert (in red)
        lbl_alert = ttk.Label(frm, textvariable=self.var_alert, foreground="#cc0000", font=("Segoe UI", 10, "bold"))
        lbl_alert.grid(row=8, column=0, columnspan=5, sticky="w", pady=(8,0))

        # Expand
        for i in range(5):
            frm.columnconfigure(i, weight=1)
        frm.rowconfigure(6, weight=1)

        self._tick()
        self._root.mainloop()

    def _tick(self):
        if not self._alive:
            try:
                self._root.destroy()
            except Exception:
                pass
            return

        with self._status_lock:
            s = dict(self._status)

        self.var_expected.set(f"Expected: {s['expected']}")
        self.var_index.set(f"Index: {s['index']+1}/{s['total']}")
        self.var_next.set(f"Next: {s['next_hint'] or '-'}")
        self.var_alert.set(f"ALERT: {s['alert']}" if s['alert'] else "")

        # Update queue list
        self.list_queue.delete(0, self.tk.END)
        for line in s["queue_preview"]:
            self.list_queue.insert(self.tk.END, line)

        # Update expected photo
        exp_id = s.get("expected_id")
        if exp_id:
            path = os.path.join("./known_faces", f"{exp_id}.png")
            if os.path.exists(path):
                try:
                    from PIL import Image
                    img = Image.open(path).resize((120, 150))
                    from PIL import ImageTk
                    self.photo_cache = ImageTk.PhotoImage(img)
                    self.photo_label.configure(image=self.photo_cache, text="")
                except Exception:
                    self.photo_label.configure(image="", text="(No image)")
            else:
                self.photo_label.configure(image="", text="(No image)")
        else:
            self.photo_label.configure(image="", text="(No image)")

        self._root.after(150, self._tick)

# MAIN CEREMONY APP
def main():
    print("Camera starting…")
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    # --- Speech system (PowerShell-based TTS; robust on Windows) ---
    speaking_now = threading.Event()
    speech_q = pyqueue.Queue()

    def tts_worker():
        import sys, subprocess
        is_windows = sys.platform.startswith('win')
        while True:
            item = speech_q.get()
            if item is None:
                break  # shutdown
            text, cb = item if isinstance(item, tuple) else (item, None)

            ok = False
            try:
                if is_windows:
                    # Escape single quotes for PowerShell
                    ps_text = (text or "").replace("'", "''")
                    ps_script = (
                        "Add-Type -AssemblyName System.Speech; "
                        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        "$s.Rate = 0; $s.Volume = 100; "
                        "$v = ($s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Name -like '*Zira*' })[0]; "
                        "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) }; "
                        f"$s.Speak('{ps_text}')"
                    )
                    # Run without spawning a console window
                    creationflags = 0
                    startupinfo = None
                    try:
                        import subprocess as sp
                        creationflags = sp.CREATE_NO_WINDOW  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    subprocess.run(
                        ["powershell", "-NoProfile", "-Command", ps_script],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=creationflags,
                        startupinfo=startupinfo,
                    )
                    ok = True
                else:
                    # Fallbacks for other OSes (no-ops if not installed)
                    try:
                        subprocess.run(["say", text], check=True)  # macOS
                        ok = True
                    except Exception:
                        try:
                            subprocess.run(["espeak", text], check=True)  # Linux
                            ok = True
                        except Exception:
                            print("[TTS] No supported TTS backend on this OS.")
                            ok = False
            except Exception as e:
                print("[TTS error]", e)
                ok = False
            finally:
                if cb:
                    try:
                        cb(success=ok)
                    except Exception:
                        pass
                speech_q.task_done()

    # Start worker once at app start
    threading.Thread(target=tts_worker, daemon=True).start()

    def enqueue_tts(text, cb=None):
        speech_q.put((text, cb))

    print("Fetch students…")
    sequence = sort_by_seat(fetch_students(only_registered=True))
    total_seq = len(sequence)
    idx = 0

    all_students = fetch_students(only_registered=None)
    by_id = {s["student_id"]: s for s in all_students}

    print("Train (feature precompute)…")
    detector = ViolaJonesDetector(minSize=(150,150))
    preproc  = FacePreprocessor()

    # --- Stage 1: LBP (fast) ---
    extractor_lbp = LBPFeatureExtractor(grid_x=7, grid_y=7)
    feature_db_lbp = build_feature_db(sequence, detector, preproc, extractor_lbp)

    # --- Stage 2: PCA (fallback; precompute so switching is instant) ---
    extractor_pca = PCAFeatureExtractor(num_components=PCA_COMPONENTS)
    feature_db_pca = build_feature_db(sequence, detector, preproc, extractor_pca)

    # --- Active extractor/DB (start with LBP) ---
    using_pca = False
    extractor = extractor_lbp
    feature_db = feature_db_lbp
    dist_thresh = LBP_DISTANCE_THRESHOLD

    gesture_sys = HandGestureRecognizer(max_hands=2)
    gesture_sys.add_gesture(OKSignGesture(threshold=23))

    from collections import deque
    queue = deque()
    current_scan = None
    waiting_ok = False
    t0 = None
    attempts = matches = 0
    distances, latencies = [], []
    alert = None
    timeout_announced = False

    # success banner
    show_match_banner_until = 0.0
    banner_text = ""
    ok_gate_open = False

    # --- Staff panel ---
    panel = StaffPanelTk(title="Graduation Staff Panel")

    def draw_instructions(frame, stage):
        x0, y0 = 15, 15
        if stage == "qr":
            cv2.putText(frame, "1) Show QR Code", (x0, y0+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,255,200), 2)
        elif stage == "ok":
            cv2.putText(frame, "2) Show 'OK' sign when ready", (x0, y0+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,255,200), 2)

    print("Ready. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]

        expected = sequence[idx] if idx < total_seq else None
        next_hint = None
        if idx+1 < total_seq:
            n = sequence[idx+1]
            next_hint = f"{n.get('seat_num','?')} | {n.get('name','?')}"

        # --- Instructions ---
        if current_scan is None:
            draw_instructions(frame, "qr")
        elif waiting_ok:
            draw_instructions(frame, "ok")

        # --- Step 1: QR ---
        if current_scan is None:
            qr = try_decode_qr(frame)
            if qr and "student_id" in qr:
                sid = qr["student_id"]

                if sid not in by_id:
                    cv2.putText(frame, "Invalid QR code. Please try again.", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                else:
                    scanned = by_id[sid]
                    if expected:
                        try:
                            exp_seat = int(expected.get("seat_num", 0))
                            got_seat = int(scanned.get("seat_num", 0))
                        except:
                            exp_seat = got_seat = -1
                        if exp_seat != got_seat:
                            alert = (f"Seat mismatch! Expected {exp_seat}, got {got_seat} "
                                     f"({scanned.get('name','?')}). Press Force to accept anyway.")
                            cv2.putText(frame, "Seat mismatch. Please wait for staff.", (20, 80),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                            # keep waiting for the correct student and ensure we’re in LBP
                            using_pca = False
                            extractor = extractor_lbp
                            feature_db = feature_db_lbp
                            dist_thresh = LBP_DISTANCE_THRESHOLD
                            t0 = None
                            waiting_ok = False
                            current_scan = None
                        else:
                            current_scan = scanned
                            waiting_ok = OK_REQUIRED
                            t0 = None
                            timeout_announced = False
                            ok_gate_open = False
                            gesture_sys.last_result = None
                    else:
                        current_scan = scanned
                        waiting_ok = OK_REQUIRED
                        t0 = None
                        timeout_announced = False
                        ok_gate_open = False
                        gesture_sys.last_result = None

        else:
            cv2.putText(frame,
                        f"Target: {current_scan.get('name','?')} ({current_scan.get('student_id','?')})",
                        (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # --- Step 2: OK gesture ---
        if current_scan is not None and waiting_ok:
            is_ok_now = detect_ok_sign(gesture_sys, frame)

            # Open the gate only after we've seen at least one non-OK frame
            if not is_ok_now:
                ok_gate_open = True

            # Accept OK only on a rising edge (non-OK -> OK)
            if is_ok_now and ok_gate_open:
                waiting_ok = False
                t0 = time.monotonic()
                timeout_announced = False

                # reset for safety
                ok_gate_open = False
                gesture_sys.last_result = None

        # --- Step 3: Face compare ---
        if current_scan is not None and not waiting_ok:
            if t0 is None:
                t0 = time.monotonic()
            elapsed = time.monotonic() - t0

            sid = current_scan.get("student_id")
            ref = feature_db.get(sid)          # uses the ACTIVE DB (LBP or PCA)
            if ref is not None:
                faces, _ = detector.detect_faces(frame)
                empty = (faces is None or len(faces) == 0)
                if not empty:
                    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

                    t_start = time.perf_counter()
                    face_img = preproc.preprocess(frame, (x, y, w, h))
                    feat = extractor.extract(face_img)  # ACTIVE extractor
                    dist = FaceRecognizer._chi2_distance(feat, ref)
                    latency_ms = (time.perf_counter() - t_start) * 1000.0

                    attempts += 1
                    distances.append(dist)
                    latencies.append(latency_ms)

                    if dist < dist_thresh:
                        # --- success ---
                        matches += 1
                        queue.append(current_scan)
                        show_match_banner_until = time.monotonic() + 2.5
                        banner_text = "Face matched. Please proceed"

                        # advance expected pointer
                        if expected:
                            if expected.get("student_id") == sid:
                                # normal case: matched the expected person
                                idx = min(idx + 1, total_seq)
                            else:
                                # (optional) realign if the matched person is later in the list
                                try:
                                    pos = next(i for i, s in enumerate(sequence)
                                            if s.get("student_id") == sid)
                                    idx = min(pos + 1, total_seq)
                                except StopIteration:
                                    # if not found in sequence, leave idx as-is
                                    pass
                        else:
                            # if there was no expected (end of list), keep idx as-is
                            pass

                        # reset state for next student (return to LBP)
                        current_scan = None
                        waiting_ok = False
                        t0 = None
                        alert = None
                        timeout_announced = False
                        using_pca = False
                        extractor = extractor_lbp
                        feature_db = feature_db_lbp
                        dist_thresh = LBP_DISTANCE_THRESHOLD
                        ok_gate_open = False
                        gesture_sys.last_result = None
                    else:
                        if (not using_pca) and (elapsed > MATCH_TIMEOUT_SEC):
                            using_pca = True
                            extractor = extractor_pca
                            feature_db = feature_db_pca
                            dist_thresh = PCA_DISTANCE_THRESHOLD
                            t0 = time.monotonic()           # keep if you still want elapsed stats
                            alert = "No match yet. Fallback to PCA (no timeout)."
                            timeout_announced = False
                            ok_gate_open = False
                            gesture_sys.last_result = None

        # --- Staff Panel commands ---
        cmd = panel.get_command_nowait()
        if cmd == "quit":
            break
        elif cmd == "skip":
            if expected:
                idx = min(idx+1, total_seq)
            # full reset & return to LBP
            current_scan = None
            waiting_ok = False
            t0 = None
            alert = None
            timeout_announced = False
            using_pca = False
            extractor = extractor_lbp
            feature_db = feature_db_lbp
            dist_thresh = LBP_DISTANCE_THRESHOLD
            ok_gate_open = False
            gesture_sys.last_result = None
        elif cmd == "force":
            target = current_scan if current_scan is not None else expected
            if target is not None:
                queue.append(target)
                show_match_banner_until = time.monotonic() + 2.5
                banner_text = "MATCH. Enqueued (Forced)"
                if expected:
                    idx = min(idx+1, total_seq)
                # full reset & return to LBP
                current_scan = None
                waiting_ok = False
                t0 = None
                alert = None
                timeout_announced = False
                using_pca = False
                extractor = extractor_lbp
                feature_db = feature_db_lbp
                dist_thresh = LBP_DISTANCE_THRESHOLD
                ok_gate_open = False
                gesture_sys.last_result = None
        elif cmd == "call":
            if queue and not speaking_now.is_set():
                called = queue[0]  # peek
                name  = called.get("name","(unknown)")
                award = called.get("award","")

                speaking_now.set()
                print(f"[CALL] {name} — {award}")

                def after_speak(success=True):
                    try:
                        if success and queue and queue[0] is called:
                            queue.popleft()
                    finally:
                        speaking_now.clear()

                enqueue_tts(f"{name}. {award}.", after_speak)

        # Keyboard fallback
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

        # HUD
        rate = (matches / attempts) if attempts else 0.0
        cv2.putText(frame, f"Attempts:{attempts}  Rate:{rate:.2f}",
                    (20, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        if time.monotonic() < show_match_banner_until:
            overlay = frame.copy()
            banner_h = 46
            cv2.rectangle(overlay, (0, H - banner_h), (W, H), (0, 200, 0), -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
            cv2.putText(frame, banner_text, (18, H - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.imshow("Graduation Face Scanner", frame)
        panel.update_status(expected, idx, total_seq, queue, alert, next_hint)

    cap.release()
    cv2.destroyAllWindows()
    panel.shutdown()

    # Stop TTS worker
    speech_q.put(None)

    def p95(arr):
        return float(np.percentile(arr, 95)) if arr else None
    print({
        "attempts": attempts,
        "avg_distance": float(np.mean(distances)) if distances else None,
        "p95_latency_ms": p95(latencies),
        "queued_remaining": len(queue)
    })

if __name__ == "__main__":
    main()
