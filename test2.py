import cv2
import numpy as np
import face_recognition
import time



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
        rotation_anglesL: list[int, float]
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

class FaceEncoder:
    '''
    Encode face for face recognition
    '''

    def __init__(self, tolerance=0.5, model="hog"):
        """
        Parameters
        ----------
        tolerance: float
            Threshold for face recognition (lower = stricter)
        model: str
            Model used for face recognition.\n
            'hog': CPU, faster, less accurate\n
            'cnn': GPU, slow on CPU, more accurate 
        """
        self.tolerance = tolerance
        self.model = model 
        self.known_encodings = []
        self.known_labels = []
    
    def add_known_face(self, img_rgb, label='Unknown') -> bool:
        '''
        Encodes a face and store it as label
        '''
        encodings = face_recognition.face_encodings(img_rgb, model=self.model)
        if len(encodings) > 0:
            self.known_encodings.append(encodings[0])
            self.known_labels.append(label)
            return True
        return False
    
    def recognize_faces(self, img_rgb, face_locations=None):
        '''
        Detect and compares faces in img_rgb against known encodings.

        Returns
        -------
        results
            A list of (label, match, distance, (top, right, bottom, left))
        '''
        if face_locations is None:
            face_locations = face_recognition.face_locations(img_rgb, model=self.model)
        
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)
        results = []

        for face_encoding, location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=self.tolerance)
            face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                match = matches[best_match_index]
                label = self.known_labels[best_match_index] if match else "Unknown"
                distance = face_distances[best_match_index]
            else:
                match = False
                label = "Unknown"
                distance = 1.0 # no match at all
            
            result = (label, match, distance, (location))
            results.append(result)

        return results



class FaceRecognitionSystem:
    def __init__(self, 
                 encoder: FaceEncoder, 
                 augmenter: ImageAugmenter | None = None,
                 resize_factor: float = 0.25,
                 process_interval: float = 0,
                 camera_index: int = 0):

       '''
        Parameters
        ----------
        encoder : FaceEncoder
            The face encoder object (manages known encodings).
        augmenter : ImageAugmenter | None
            Optional augmenter for training images.
        resize_factor : float
            Downscaling factor for faster recognition.
        process_interval : float
            Seconds between recognition runs.
        camera_index : int
            Index of webcam (0=default).
       '''
       self.encoder = encoder
       self.augmenter = augmenter
       self.resize_factor = resize_factor
       self.process_interval = process_interval
       self.camera_index = camera_index
       
    
    def load_image(self, image_path: str, label: str):
        '''
        Load an image, apply augmentation, and stores encodings
        '''
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Image with path {image_path} not found")
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        images = [image_rgb]
        
        if self.augmenter:
            images = self.augmenter.augment(image_rgb)
        
        for img in images:
            self.encoder.add_known_face(img, label=label)
    
    def draw_results(self, frame, results, fps_display):
        for label, match, dist, (top, right, bottom, left) in results:
            color = (0, 255, 0) if match else (0, 0, 255)
            text = f"{label} ({dist:.2f})"
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, text, (left, top - 10),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        cv2.putText(frame, f"FPS: {fps_display:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    def run(self):
        video = cv2.VideoCapture(self.camera_index)
        
        last_process_time = 0
        prev_time = time.time()
        fps_display = 0
        last_fps_update = time.time()
        
        last_results = []
        
        while True:
            ret, frame = video.read()
            if not ret:
                break
            
            display_frame = frame.copy()
            
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            if current_time - last_fps_update >= 1.0:
                fps_display = fps
                last_fps_update = current_time
            
            if current_time - last_process_time >= self.process_interval:
                last_process_time = current_time
                
                small_frame = cv2.resize(frame, (0,0), fx=self.resize_factor, fy=self.resize_factor)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_small_frame, model=self.encoder.model)
                
                scaled_locations = [
                    (int(top / self.resize_factor), int(right / self.resize_factor),
                     int(bottom / self.resize_factor), int(left / self.resize_factor))
                    for (top, right, bottom, left) in face_locations
                ]
                
                last_results = []
                results = self.encoder.recognize_faces(rgb_small_frame, face_locations)
                for (label, match, dist, _), scaled_loc in zip(results, scaled_locations):
                    last_results.append((label, match, dist, scaled_loc))
                
            self.draw_results(display_frame, last_results, fps_display)
            
            cv2.imshow("Camera", display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        video.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    augmenter = ImageAugmenter()
    encoder = FaceEncoder(tolerance=0.5, model='hog')
    system = FaceRecognitionSystem(encoder, augmenter, camera_index=1)
    
    students = [
        {
            "seat_num": "1",
            "student_id": "24WMR09274",
            "name": "Tan Jee Cheng",
            "course": "Bachelor of Software Engineering",
            "award": "Graduated with Distinction",
            "image_path": "known_faces/24WMR09274.png"
        },
        {
            "seat_num": "2",
            "student_id": "24WMR09155",
            "name": "Cheong Jau Chun",
            "course": "Bachelor of Computer Science",
            "award": "Dean's List",
            "image_path": "known_faces/24WMR09155.jpg"
        },
    ]
    
    for student in students:
        system.load_image(f"./{student['image_path']}", label=student['name'])
    
    system.run()