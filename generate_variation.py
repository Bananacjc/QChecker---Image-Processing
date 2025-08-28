import cv2
import numpy as np
import face_recognition

def augment_image(img):
    augmented = []

    # Original encoding
    augmented.append(img)

    # Brightness variations
    for beta in [-40, -20, 20, 40]:  # brightness shift
        bright = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
        augmented.append(bright)

    # Contrast variations
    for alpha in [0.8, 1.2]:  # contrast scale
        contrast = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        augmented.append(contrast)

    # Rotations
    h, w = img.shape[:2]
    for angle in [-15, 15]:  # small tilts
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (w, h))
        augmented.append(rotated)

    return augmented

# Load the single provided image
image_path = './known_faces/24WMR09155.jpg'
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Generate variations
variations = augment_image(image_rgb)

# Encode all variations
encodings = []
for var_img in variations:
    face_enc = face_recognition.face_encodings(var_img)
    if len(face_enc) > 0:
        encodings.append(face_enc[0])

print(f"Generated {len(encodings)} encodings for this user.")
