import cv2
import numpy as np
from pathlib import Path
import shutil

class FaceProcessor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    def extract_faces(self, image_path, output_dir, name):
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        face_paths = []
        for i, (x, y, w, h) in enumerate(faces):
            face = image[y:y+h, x:x+w]
            output_path = Path(output_dir) / name / f"{Path(image_path).stem}_face_{i}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), face)
            face_paths.append(output_path)
        
       
        self.train_model(output_dir)
        return face_paths

    def train_model(self, training_dir):
        faces = []
        labels = []
        names = {}
        current_id = 0
        
        for person_dir in Path(training_dir).iterdir():
            if not person_dir.is_dir():
                continue
                
            name = person_dir.name
            names[current_id] = name
            
            for image_path in person_dir.glob('*.jpg'):
                image = cv2.imread(str(image_path))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces.append(gray)
                labels.append(current_id)
            
            current_id += 1
        
        if faces:
            self.recognizer.train(faces, np.array(labels))
            self.recognizer.save('recognizer.yml')
            
            with open('names.txt', 'w') as f:
                for id_, name in names.items():
                    f.write(f"{id_},{name}\n")

    def process_registration_image(self, image_path, name):
        return self.extract_faces(image_path, 'training_data', name)