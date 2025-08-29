import cv2
import numpy as np
from abc import ABC, abstractmethod
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
import os
import json

'''
Stage 1: Face Detection (Skin Color Segmentation + Morphology)
Goal: Find where faces are in the image.
Convert to color space
    Use YCbCr or HSV (better for skin than RGB).
    Typical thresholds for skin in YCbCr:
    77 ≤ Cb ≤ 127
    133 ≤ Cr ≤ 173
Thresholding
    Create a binary mask: pixels inside skin range → 1, else 0.
    Morphological Operations
    Apply opening (erosion → dilation) to remove small noise.
    Apply closing (dilation → erosion) to fill small holes.
Connected Components
    Find blobs in the mask.
    Extract bounding boxes.
    Filter by aspect ratio (1:1 to 1:1.5) and minimum size.
👉 Output: Coordinates of candidate face regions.

Stage 2: Face Preprocessing
Goal: Normalize faces before recognition.
Crop the detected bounding box.
Resize to a fixed size (e.g., 100×100).
Convert to grayscale (faces are usually recognized by structure, not color).
Normalize pixel values (e.g., scale to [0,1] or mean=0, std=1).
👉 Output: Standardized face images (same size, same format).

Stage 3: Feature Extraction (Recognition)
Choose one of the classical algorithms:
Option A: Eigenfaces (PCA)
Collect training faces for each person.
Flatten each face (100×100 → 10,000 vector).
Apply PCA to reduce dimensionality (e.g., keep top 50–100 components).
Each face is represented in Eigenface space.

Option B: Fisherfaces (LDA)
Similar to PCA, but maximizes separation between classes (better recognition).

Option C: Local Binary Patterns (LBP)
Divide face into small regions (e.g., 8×8 grid).
For each pixel: compare neighbors → build binary number.
Build histogram of LBP values per region.
Concatenate histograms into one feature vector.

👉 Recommendation: Start with Eigenfaces (PCA) (simpler, lots of tutorials), then try LBP for robustness.

Stage 4: Classification / Recognition
Store feature vectors of known faces in a database.
For a new face:
Extract its features (PCA projection / LBP histogram).
Compare with database using Euclidean distance or cosine similarity.
Pick the closest match (if distance < threshold).
Otherwise → "Unknown person".

Stage 5: Evaluation & Improvements
Test on a dataset (even your own images).
Adjust thresholds for detection & recognition.
Improve robustness with:
Illumination normalization (histogram equalization).
Face alignment (eyes aligned before recognition).
'''

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

class FacePreprocessor:
    def __init__(self, size=(100, 100), normalize=True, debug=False):
        self.size = size
        self.normalize = normalize
        self.debug = debug
        
    def preprocess(self, frame, face_box):
        x, y, w, h = face_box
        
        face = frame[y:y+h, x:x+w]
        
        face = cv2.resize(face, self.size)
        
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
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
    Simplest
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

class LBFFeatureExtractor:
    def __init__(self, grid_x=8, grid_y=8):
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
        return np.array(features)

class FaceRecognizer:
    def __init__(self, extractor: FeatureExtractor = None):
        self.extractor = extractor if extractor else PCAFeatureExtractor()
        self.database = []  # list of (feature_vector, label)
        self._trained = False
        self.threshold = None
        self.distance_func = None

    def fit(self, faces, labels=None):
        # Fit PCA or LDA depending on extractor type
        if isinstance(self.extractor, PCAFeatureExtractor):
            self.extractor.fit(faces)
            self.distance_func = self._euclidean
        elif isinstance(self.extractor, LDAFeatureExtractor) or isinstance(self.extractor, FastLDAFeatureExtractor):
            self.extractor.fit(faces, labels)
            self.distance_func = self._euclidean
        elif isinstance(self.extractor, LBFFeatureExtractor):
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
            self.threshold = max(intra_dists) * 1.2  # allow small margin
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
                        
                    sid = self.recognizer.recognize(face_img)
                    
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

if __name__ == '__main__':
    
    students = fetch_students(only_registered=True)
    
    
    faceExtractor = LBFFeatureExtractor()
    faceDetector = SkinSegmentationDetector()
    system = FacePipeline(students, extractor=faceExtractor, detector=faceDetector)
   
    system.run(camera_index=0)
    