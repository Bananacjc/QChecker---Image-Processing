import qrcode
import json
import os

# Folder to save QR codes
output_folder = "student_qrcodes"
os.makedirs(output_folder, exist_ok=True)

# Sample student list
students = [
    {
        "seat_num": "1",
        "student_id": "24WMR09274",
        "name": "Tan Jee Cheng",
        "course": "Bachelor of Software Engineering",
        "award": "Graduated with Distinction",
        "image_path": "students_face/24WMR09274.jpg"
    },
]

# Generate QR code for each student
for student in students:
    # Convert student data to JSON string
    student_data = json.dumps(student)
    
    # Generate QR
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # High correction level
        box_size=10,
        border=4,
    )
    qr.add_data(student_data)
    qr.make(fit=True)
    
    # Create image
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save image using student ID or name
    filename = f"{student['student_id']}.png"
    filepath = os.path.join(output_folder, filename)
    img.save(filepath)

    print(f"Saved QR for {student['name']} -> {filepath}")
