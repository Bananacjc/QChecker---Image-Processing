import cv2
import numpy as np
import face_recognition
import time

def augment_image(img, brightness_shift=None, constrast_scales=None, rotation_angles=None):
    if brightness_shift is None:
        brightness_shift = [-20, -10, 10, 20]
    if constrast_scales is None:
        constrast_scales = [0.8, 1.2]
    if rotation_angles is None:
        rotation_angles = [-15, 15]
    
    augmented = [img]

     # Brightness variations
    for beta in [-40, -20, 20, 40]:
        bright = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
        augmented.append(bright)

    # Contrast variations
    for alpha in [0.8, 1.2]:
        contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        augmented.append(contrast)

    # Small rotations
    h, w = img.shape[:2]
    for angle in [-15, 15]:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h))
        augmented.append(rotated)

    return augmented


def encode_face(image_path):
    """
    Load one known face image, generate variations, and return encodings.
    """

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    variations = augment_image(image_rgb)

    known_encodings = []
    for var_img in variations:
        face_enc = face_recognition.face_encodings(var_img)
        if len(face_enc) > 0:
            known_encodings.append(face_enc[0])

    print(f"Generated {len(known_encodings)} encodings for this user.")
    return known_encodings

def recognize_faces(frame, known_encodings, resize_factor=0.21, tolerance=0.5):
    """
    Face detection and recognition
    """

    small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    scaled_locations = [
        (int(top / resize_factor), int(right / resize_factor),
         int(bottom / resize_factor), int(left / resize_factor))
        for (top, right, bottom, left) in face_locations
    ]

    matches_list = []
    distances_list = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=tolerance)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

        if best_match_index is not None:
            match = matches[best_match_index]
            distance = face_distances[best_match_index]
        else:
            match = False
            distance = 1.0

        matches_list.append(match)
        distances_list.append(distance)

    return scaled_locations, matches_list, distances_list

def draw_results(frame, face_locations, matches, distances, fps_display):
    """
    Draw rectangles, match results, and FPS on the frame.
    """
    for (top, right, bottom, left), match, dist in zip(face_locations, matches, distances):
        color = (0, 255, 0) if match else (0, 0, 255)
        match_text = f"Match: {match}, Dist: {dist:.2f}"
        print(match_text)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, match_text, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(frame, f"FPS: {fps_display:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

def run_realtime_recognition(known_encodings):
    video = cv2.VideoCapture(1)

    resize_factor = 0.21
    last_process_time = 0
    process_interval_sec = 1  # Process every 1 second

    prev_time = time.time()
    fps_display = 0
    last_fps_update = time.time()

    last_face_locations = []
    last_face_matches = []
    last_face_distances = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        display_frame = frame.copy()

        # FPS calculation
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        if current_time - last_fps_update >= 1.0:
            fps_display = fps
            last_fps_update = current_time

        # Process recognition every second
        if current_time - last_process_time >= process_interval_sec:
            last_process_time = current_time
            last_face_locations, last_face_matches, last_face_distances = recognize_faces(
                frame, known_encodings, resize_factor=resize_factor
            )

        # Draw results
        draw_results(display_frame, last_face_locations, last_face_matches, last_face_distances, fps_display)

        cv2.imshow("Camera", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    encodings = encode_face("./known_faces/24WMR09155.jpg")
    run_realtime_recognition(encodings)
