from flask import Flask, render_template, request, jsonify, send_file, Response, g
from werkzeug.utils import secure_filename
import sqlite3
import pandas as pd
from pathlib import Path
import os
import json
import cv2
import numpy as np
from datetime import datetime
import logging

# App Configuration and Initialization
class SmartPresenceApp:
    def __init__(self):
        self.app = Flask(__name__)
        
        # App Configuration
        self.app.config.update(
            FACE_RECOGNITION_THRESHOLD=70,
            UPLOAD_FOLDER='uploads',
            MAX_CONTENT_LENGTH=16 * 1024 * 1024,
            DATABASE='attendance.db',
            SECRET_KEY='your-secret-key-here'
        )
        
        # Setup Logging
        self.setup_logging()
        
        # Create Required Directories
        self.create_directories()
        
        # Register Routes
        self.register_routes()
        
        # Initialize Database
        self.init_db()
    
    def setup_logging(self):
        """Configure logging for the application"""
        logging.basicConfig(
            filename='smartpresence.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SmartPresence')
    
    def create_directories(self):
        """Create necessary directories for the application"""
        directories = [
            'uploads', 
            'training_data', 
            'static/exports', 
            'static/faces'
        ]
        for dir_name in directories:
            Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    def register_routes(self):
        """Register all application routes"""
        self.app.add_url_rule('/', 'index', self.index)
        self.app.add_url_rule('/register', 'register', self.register, methods=['GET', 'POST'])
        self.app.add_url_rule('/video_feed', 'video_feed', self.video_feed)
        self.app.add_url_rule('/live', 'live', self.live)
        self.app.add_url_rule('/attendance', 'attendance', self.attendance)
        self.app.add_url_rule('/reports', 'reports', self.reports)
        self.app.add_url_rule('/export', 'export', self.export)
        self.app.add_url_rule('/debug_database', 'debug_database', self.debug_database)
    
    def init_db(self):
        """Initialize the database with the correct schema"""
        try:
            with sqlite3.connect(self.app.config['DATABASE']) as db:
                db.execute('''
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL DEFAULT 0,
                    distance REAL DEFAULT 0
                )
                ''')
                db.commit()
            self.migrate_database()
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization error: {e}")
    
    def migrate_database(self):
        """Migrate database schema if needed"""
        try:
            conn = sqlite3.connect(self.app.config['DATABASE'])
            cursor = conn.cursor()
            
            # Check existing columns
            cursor.execute("PRAGMA table_info(attendance)")
            columns = [column[1] for column in cursor.fetchall()]
            
            # Migration steps
            migration_queries = [
                ('distance', 'ALTER TABLE attendance ADD COLUMN distance REAL DEFAULT 0'),
                ('confidence', 'ALTER TABLE attendance ADD COLUMN confidence REAL DEFAULT 0')
            ]
            
            for column, query in migration_queries:
                if column not in columns:
                    try:
                        cursor.execute(query)
                        self.logger.info(f"Added '{column}' column to attendance table")
                    except sqlite3.Error as e:
                        self.logger.warning(f"Could not add {column} column: {e}")
            
            conn.commit()
            conn.close()
        except sqlite3.Error as e:
            self.logger.error(f"Database migration error: {e}")
    
    def get_db(self):
        """Create a database connection"""
        if not hasattr(g, 'sqlite_db'):
            g.sqlite_db = sqlite3.connect(self.app.config['DATABASE'])
            g.sqlite_db.row_factory = sqlite3.Row
        return g.sqlite_db
    
    def close_db(self, error):
        """Close the database connection"""
        if hasattr(g, 'sqlite_db'):
            g.sqlite_db.close()
    
    def debug_database(self):
        """Debug route to check database schema"""
        try:
            conn = sqlite3.connect(self.app.config['DATABASE'])
            cursor = conn.cursor()
            
            # Check table information
            cursor.execute("PRAGMA table_info(attendance)")
            columns = cursor.fetchall()
            
            debug_info = []
            for column in columns:
                debug_info.append({
                    'name': column[1],
                    'type': column[2],
                    'nullable': column[3],
                    'default': column[4],
                    'primary_key': column[5]
                })
            
            conn.close()
            return jsonify(debug_info)
        except sqlite3.Error as e:
            return f"Database schema check error: {e}", 500

    # Route Handlers
    def index(self):
        """Render the index page"""
        return render_template('index.html')
    
    def register(self):
        """Handle user registration"""
        if request.method == 'POST':
            try:
                if 'photo' not in request.files:
                    return jsonify({'error': 'No photo uploaded'})
                
                photo = request.files['photo']
                name = request.form['name']
                
                if photo.filename == '':
                    return jsonify({'error': 'No photo selected'})
                
                filename = secure_filename(photo.filename)
                photo_path = Path(self.app.config['UPLOAD_FOLDER']) / filename
                photo.save(str(photo_path))
                
                processor = FaceProcessor()
                face_paths = processor.process_registration_image(photo_path, name)
                
                photo_path.unlink()
                
                return jsonify({
                    'success': True,
                    'message': f'Registered {len(face_paths)} faces for {name}'
                })
            except Exception as e:
                self.logger.error(f"Registration error: {e}")
                return jsonify({'error': str(e)})
        
        return render_template('register.html')
    
    def video_feed(self):
        """Generate video feed for live recognition"""
        return Response(gen(VideoCamera()),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    def live(self):
        """Render live recognition page"""
        return render_template('live.html')
    
    def attendance(self):
        """Fetch and display attendance records"""
        try:
            db = self.get_db()
            cursor = db.cursor()
            
            # Fetch records
            cursor.execute('''
                SELECT name, timestamp, 
                       COALESCE(confidence, 0) as confidence, 
                       COALESCE(distance, 0) as distance
                FROM attendance 
                ORDER BY timestamp DESC 
                LIMIT 100
            ''')
            records = cursor.fetchall()
            
            # Convert to list of dictionaries
            records = [dict(record) for record in records]
            
            return render_template('attendance.html', records=records)
        
        except sqlite3.Error as e:
            self.logger.error(f"SQLite error in attendance route: {e}")
            return f"Database error: {e}", 500
        except Exception as e:
            self.logger.error(f"Unexpected error in attendance route: {e}")
            return f"Unexpected error: {e}", 500
    
    def reports(self):
        """Generate attendance reports"""
        try:
            db = self.get_db()
            daily = pd.read_sql_query('''
                SELECT name, COUNT(*) as count, 
                       AVG(COALESCE(confidence, 0)) as avg_confidence,
                       AVG(COALESCE(distance, 0)) as avg_distance,
                       MAX(timestamp) as last_seen
                FROM attendance
                WHERE date(timestamp) = date('now')
                GROUP BY name
            ''', db)
            
            monthly = pd.read_sql_query('''
                SELECT 
                    name,
                    strftime('%Y-%m', timestamp) as month,
                    COUNT(*) as count,
                    AVG(COALESCE(confidence, 0)) as avg_confidence
                FROM attendance
                GROUP BY name, month
                ORDER BY month DESC, name
            ''', db)
            
            return render_template('reports.html', 
                                 daily=daily.to_dict('records'),
                                 monthly=monthly.to_dict('records'))
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
            return "Error generating reports", 500
    
    def export(self):
        """Export attendance records to Excel"""
        try:
            db = self.get_db()
            df = pd.read_sql_query('SELECT * FROM attendance', db)
            excel_path = 'static/exports/attendance.xlsx'
            df.to_excel(excel_path, index=False)
            return send_file(excel_path, as_attachment=True)
        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return "Error exporting data", 500

class FaceProcessor:
    def __init__(self):
        """Initialize face processing components"""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    def extract_faces(self, image_path, output_dir, name):
        """Extract faces from an image"""
        image = cv2.imread(str(image_path))
        if image is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20),
            maxSize=(500, 500)
        )
        
        face_paths = []
        for i, (x, y, w, h) in enumerate(faces):
            face = image[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))
            output_path = Path(output_dir) / name / f"{Path(image_path).stem}_face_{i}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), face)
            face_paths.append(output_path)
        
        return face_paths

    def train_model(self, training_dir):
        """Train face recognition model"""
        faces = []
        labels = []
        names = {}
        current_id = 0
        
        for person_dir in Path(training_dir).iterdir():
            if not person_dir.is_dir():
                continue
                
            name = person_dir.name
            names[current_id] = name
            logging.info(f"Training on {name}'s images")
            
            for image_path in person_dir.glob('*.jpg'):
                image = cv2.imread(str(image_path))
                if image is not None:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    gray = cv2.equalizeHist(gray)
                    faces.append(gray)
                    labels.append(current_id)
            
            current_id += 1
        
        if faces:
            logging.info(f"Training model with {len(faces)} images")
            self.recognizer.train(faces, np.array(labels))
            self.recognizer.save('recognizer.yml')
            
            with open('names.txt', 'w') as f:
                for id_, name in names.items():
                    f.write(f"{id_},{name}\n")
            
            logging.info("Model training completed")
            return True
        return False

    def process_registration_image(self, image_path, name):
        """Process registration image and train model"""
        faces = self.extract_faces(image_path, 'training_data', name)
        if faces:
            self.train_model('training_data')
        return faces

class VideoCamera:
    def __init__(self):
        """Initialize video camera for face recognition"""
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        self.detection_params = {
            'scaleFactor': 1.05,
            'minNeighbors': 3,
            'minSize': (20, 20),
            'maxSize': (500, 500)
        }
        
        try:
            self.recognizer.read('recognizer.yml')
            self.names = {}
            with open('names.txt', 'r') as f:
                for line in f:
                    id_, name = line.strip().split(',')
                    self.names[int(id_)] = name
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.warning(f"No trained model found: {e}")
            self.names = {}

    def __del__(self):
        """Release video capture on object deletion"""
        self.video.release()

    def estimate_distance(self, face_width):
        """Estimate distance based on face width"""
        KNOWN_WIDTH = 16.0
        KNOWN_DISTANCE = 50.0
        KNOWN_PIXEL_WIDTH = 200.0
        distance = (KNOWN_WIDTH * KNOWN_DISTANCE) / (face_width / KNOWN_PIXEL_WIDTH)
        return distance / 100

    def get_frame(self):
        """Capture and process video frame"""
        success, frame = self.video.read()
        if not success:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.detection_params['scaleFactor'],
            minNeighbors=self.detection_params['minNeighbors'],
            minSize=self.detection_params['minSize'],
            maxSize=self.detection_params['maxSize']
        )
        
        # Face recognition and drawing
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (200, 200))
            
            # Predict the face
            try:
                label, confidence = self.recognizer.predict(roi_gray)
                name = self.names.get(label, 'Unknown')
                
                # Determine confidence and distance
                confidence_score = 100 - confidence
                face_width = w
                estimated_distance = self.estimate_distance(face_width)
                
                # Draw rectangle and label
                color = (0, 255, 0) if confidence_score > 70 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Text for name, confidence, and distance
                text = f"{name} ({confidence_score:.2f}%)"
                cv2.putText(frame, text, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Log and save attendance if confident enough
                if confidence_score > 70:
                    self.log_attendance(name, confidence_score, estimated_distance)
                
            except Exception as e:
                logging.error(f"Face recognition error: {e}")
        
        # Encode frame for streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def log_attendance(self, name, confidence, distance):
        """Log attendance to database"""
        try:
            # This method needs to be adapted to match the application's database setup
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO attendance (name, confidence, distance) 
                VALUES (?, ?, ?)
            ''', (name, confidence, distance))
            
            conn.commit()
            conn.close()
            
            logging.info(f"Logged attendance for {name}")
        except sqlite3.Error as e:
            logging.error(f"Error logging attendance: {e}")

def gen(camera):
    """Generator function for video streaming"""
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# Application Initialization and Running
def run_smart_presence_app():
    """Run the Smart Presence Application"""
    app_instance = SmartPresenceApp()
    
    @app_instance.app.teardown_appcontext
    def close_db(error):
        """Close database at the end of the request"""
        app_instance.close_db(error)
    
    # Optional: Add error handlers
    @app_instance.app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404
    
    @app_instance.app.errorhandler(500)
    def internal_server_error(e):
        return render_template('500.html'), 500
    
    # Run the app
    return app_instance.app

# Main execution block
if __name__ == '__main__':
    smart_presence_app = run_smart_presence_app()
    
    # Configuration for development
    smart_presence_app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=True,
        threaded=True
    )