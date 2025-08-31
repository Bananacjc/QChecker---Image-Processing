import qrcode
import json
import os

class QRGenerator():
    def __init__(self, students: dict[str, str]):
        self.students = json.dumps(students)
        self.generateQR()
    
    def generateQR(self):
        for student in self.students:
            
            folder_path = "student_QR"
            filename = f"{student['student_id']}_QR.png"
            
            qr = qrcode.QRCode(version=1, error_correction=qrcode.ERROR_CORRECT_H,
                            box_size=10, border=4)
            qr.add_data(student)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="blue", back_color="white")
            
        
            self.saveToFolder(folder_path, filename, img)
           
    
    def saveToFolder(self, folder_path: str, filename: str, img):
        filepath = os.path.join(folder_path, filename)
        img.save(filepath)
        
if __name__ == '__main__':
    students = [
        {
            "seat_num": "1",
            "student_id": "24WMR09274",
            "name": "Tan Jee Cheng",
            "course": "Bachelor of Software Engineering",
            "award": "Graduated with Distinction",
        },
        {
            "seat_num": "2",
            "student_id": "24WMR09155",
            "name": "Cheong Jau Chun",
            "course": "Bachelor of Software Engineering",
            "award": "Graduated with Merit",
        }
    ]