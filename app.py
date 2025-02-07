# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, session
import pymysql
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
import pandas as pd
import re
import threading
import csv
import time as time_module  # Rename time module to avoid conflict
from functools import wraps
from datetime import datetime, timedelta
import time as time




# Flask app setup
app = Flask(__name__)
app.secret_key = "3d6f45a5fc12445dbac2f59c3b6c7cb1"
HAARCASCADE_PATH = 'haarcascade_frontalface_default.xml'

# Database connection
def get_db_connection():
    return pymysql.connect(host='localhost', user='root', password='', db='FRAS')

# Function to check if STUDENT EXISTS
def student_exists(SID: str, Name: str) -> bool:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM StudentDetails WHERE SID = %s OR Name = %s", (SID, Name))
        result = cursor.fetchone()
        conn.close()
        return result is not None
    except Exception as e:
        print(f"Error in student_exists: {e}")
        return False
    
# Video feed generator with face detection
captured_images = 0  # Global tracker for captured images


def generate_capture_feed(SID, Name):
    global captured_images

    cam = cv2.VideoCapture(0)
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if face_classifier.empty():
        print("Error loading Haarcascade file!")
        return

    # Ensure output folder exists
    output_folder = "TrainingImage"
    os.makedirs(output_folder, exist_ok=True)

    # Initialize variables for progress tracking
    sampleNum = 0
    captured_images = 0  # Reset global counter
    start_time = time.time()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw rectangle on the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Save the detected face image
            sampleNum += 1
            face = gray[y:y + h, x:x + w]
            resized_face = cv2.resize(face, (1280, 1024))
            image_path = os.path.join(output_folder, f"{Name}.{SID}.{sampleNum}.jpg")
            cv2.imwrite(image_path, resized_face)
            print(f"Captured image saved at: {image_path}")

            captured_images += 1  # Update global counter

        # Encode frame for live streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Stop after capturing 20 images or 10 seconds
        if captured_images >= 20 or time.time() - start_time > 10:
            break

    cam.release()


def register_student(ID: int, SID: str, Name: str):
    try:
        connection = pymysql.connect(host='localhost', user='root', password='', db='FRAS')
        cursor = connection.cursor()
        
        # Insert new student into the database
        ts = datetime.now()
        Date = ts.strftime('%Y-%m-%d')
        Time = ts.strftime('%H:%M:%S')
        
        cursor.execute(
            "INSERT INTO StudentDetails (ID, SID, Name, Date, Time) VALUES (%s, %s, %s, %s, %s)",
            ( ID, SID, Name, Date, Time)
        )
        connection.commit()
        
        print(f"✅ Student {Name} (SID: {SID}) registered successfully.")
    except Exception as e:
        print(f"❌ Error registering student: {e}")

def get_highest_image_number(SID, Name, folder="TrainingImage"):
    max_number = 0
    try:
        # Get all files in the folder
        files = os.listdir(folder)
        pattern = rf"{re.escape(Name)}\.{re.escape(SID)}\.(\d+)\.jpg"
        
        for file in files:
            if file == 'TrainingImage/.DS_Store':
                pass
            else:
                match = re.match(pattern, file)
                if match:
                    number = int(match.group(1))
                    if number > max_number:
                        max_number = number
    except Exception as e:
        print(f"Error getting highest image number: {e}")
    
    return max_number


#Function to capture image (REGISTER)
def capture_images(SID, Name):



    ID = int(re.search(r'\d+', SID).group())
    register_student(ID, SID, Name)

    esp32_stream_url = "http://172.20.10.14:81/stream"
    
    cam = cv2.VideoCapture(0)
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if face_classifier.empty():
        print("Error loading Haarcascade file!")
        return

    sampleNum = 0
    output_folder = "TrainingImage"
    os.makedirs(output_folder, exist_ok=True)

    # Timer to control the interval between captures
    last_capture_time = time.time()

    while sampleNum < 20:  # Capture 20 images
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        current_time = time.time()

        for (x, y, w, h) in faces:
            # Save images after x amount of time
            if current_time - last_capture_time >= 0.5:
                sampleNum += 1
                face = gray[y:y + h, x:x + w]
                resized_face = cv2.resize(face, (1280, 1024))
                image_path = os.path.join(output_folder, f"{Name}.{SID}.{sampleNum}.jpg")
                cv2.imwrite(image_path, resized_face)
                print(f"Captured image {sampleNum} saved at: {image_path}")
                
                # Update the last capture time
                last_capture_time = current_time

            # Draw a rectangle on the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Yield the live video feed
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Stop if we've captured the target number of images
        if sampleNum >= 20:
            break

    cam.release()



#Function to capture image (REGISTER)
def capture_images_addon(SID, Name):

    ID = int(re.search(r'\d+', SID).group())

    esp32_stream_url = "http://172.20.10.14:81/stream"
    cam = cv2.VideoCapture(0)
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if face_classifier.empty():
        print("Error loading Haarcascade file!")
        return None

    # Get the highest image number for the user
    highest_number = get_highest_image_number(SID, Name)
    sampleNum = highest_number  # Start from the next sequence number

    output_folder = "TrainingImage"
    os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to capture image")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

        for (x, y, w, h) in faces:
            sampleNum += 1
            face = gray[y:y+h, x:x+w]
            resized_face = cv2.resize(face, (256, 256))  

             # Save the resized face
            image_path = os.path.join(output_folder, f"{Name}.{SID}.{sampleNum}.jpg")
            cv2.imwrite(image_path, resized_face)

            print(f"Captured image saved at: {image_path}")

        # Break after capturing images (you can increase this limit)
        if sampleNum >= highest_number + 40:
            break

    cam.release()
    return output_folder

# not used
def insert_images_to_db(SID: str, Name: str):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        path = 'static/TrainingImage'
        image_paths = [os.path.join(path, f) for f in os.listdir(path) if SID in f and Name in f]
        
        for image_path in image_paths:
            with open(image_path, 'rb') as file:
                binary_data = file.read()
                cursor.execute("INSERT INTO StudentImages (SID, Image) VALUES (%s, %s)", (SID, binary_data))
        
        conn.commit()
        conn.close()
        return f"Images for {Name} (SID: {SID}) successfully inserted."
    except Exception as e:
        return f"Error inserting images: {e}"
    
# Video feed generator with face detection
def generate_capture_feed(SID, Name):
    cam = cv2.VideoCapture(0)
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    if face_classifier.empty():
        print("Error loading Haarcascade file!")
        return

    start_time = time.time()
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if time.time() - start_time > 10:
            break

    cam.release()

# Face recognition function for live feed
attendance_records = []

def take_attendance(subject, section):
    try:
        # Get SubjectID for the subject code
        connection = get_db_connection()
        cursor = connection.cursor(pymysql.cursors.DictCursor)
        
        # Fetch SubjectID from Subjects table
        cursor.execute("SELECT * FROM Subjects WHERE SubjectCode = %s", (subject,))
        subject_data = cursor.fetchone()
        if not subject_data:
            print(f"Error: Subject {subject} not found.")
            return "Error: Subject not found."
        
        subject_id = subject_data['SubjectID']
        subject_code = subject_data['SubjectCode']

        # Check if within allowed schedule
        is_allowed = is_within_schedule(subject_code, section)
        if not is_allowed:  
            print("Attendance taking is not allowed at this time.")
            return "Error: Attendance not allowed at this time."

        # Face recognition and attendance logic starts here
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("TrainingImageLabel\Trainner.yml")

        face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        # Fetch student data along with section
        cursor.execute("""
        SELECT sd.ID, sd.SID AS StudentID, sd.Name, ss.Section, ss.SubjectID
        FROM StudentDetails sd
        INNER JOIN Student_has_Subject ss ON sd.ID = ss.StudentID
        WHERE ss.SubjectID = %s
        """, (subject_id,))

        student_data = cursor.fetchall()
        student_df = pd.DataFrame(student_data)

        # Debug: Print DataFrame columns
        print("Columns in student_df:", student_df.columns)

        # ESP, conf = 100 | cam laptop, conf = 70
        esp32_stream_url = "http://172.20.10.14:81/stream"
        cam = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        seen_students = set()

        start_time = time.time()
        while True:
            ret, frame = cam.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(50, 50))

            for (x, y, w, h) in faces:
                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
                print(f"Detected face. Predicted ID: {id}, Confidence: {confidence}")

                if confidence < 70:
                    id = str(id).strip()
                    student_df['ID'] = student_df['ID'].astype(str).str.strip()
                    matched_student = student_df.loc[student_df['ID'] == id]

                    if not matched_student.empty:
                        student_id = matched_student.iloc[0]['ID']
                        sid = matched_student.iloc[0]['StudentID']
                        name = matched_student.iloc[0]['Name']
                        student_section = matched_student.iloc[0]['Section']
                        subject_id = matched_student.iloc[0]['SubjectID']
                        date = datetime.now().strftime('%Y-%m-%d')
                        time_stamp = datetime.now().strftime('%H:%M:%S')

                        # Check if the student is in the selected section
                        if student_section == section:
                            label = f"{sid} - {name} ({section})"
                            attended = True
                        else:
                            label = f"{sid} - {name} (Not In Section)"
                            attended = False

                        # Ensure no duplicate attendance
                        if student_id not in seen_students:
                            seen_students.add(student_id)
                            attendance_records.append({
                                "ID": student_id,
                                "SID": sid,
                                "Name": name,
                                "SubjectID": subject_id,
                                "Section": section,
                                "Date": date,
                                "Time": time_stamp,
                                "Attended": attended
                            })

                        cv2.putText(frame, label, (x, y - 10), font, 0.75, (255, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Not Registered", (x, y - 10), font, 0.75, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y - 10), font, 0.75, (0, 0, 255), 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # Render the frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame_data = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

            if time.time() - start_time > 9999999:  # Stop after 30 seconds
                break

        cam.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error in recognition feed: {e}")



def is_within_schedule(subject_id, section):
    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    try:
        # Get current time and day
        current_time = datetime.now().time()  # Current time as a `time` object
        current_day = datetime.now().strftime('%A')  # e.g., 'Monday'

        # Debugging: Log current time and day
        print(f"Current time: {current_time}, Current day: {current_day}")

        # Query the schedule
        cursor.execute("""
            SELECT StartTime, EndTime
            FROM SubjectAttendanceSchedule
            WHERE SubjectID = %s AND Section = %s AND DayOfWeek = %s
        """, (subject_id, section, current_day))
        schedule = cursor.fetchone()

        if not schedule:
            print(f"No schedule found for SubjectID: {subject_id}, Section: {section}, Day: {current_day}")
            return False

        # Convert StartTime and EndTime to `time` objects if they are `timedelta`
        start_time = (
            (datetime.min + schedule['StartTime']).time()
            if isinstance(schedule['StartTime'], timedelta) else schedule['StartTime']
        )
        end_time = (
            (datetime.min + schedule['EndTime']).time()
            if isinstance(schedule['EndTime'], timedelta) else schedule['EndTime']
        )

        # Debugging: Print the fetched schedule
        print(f"Schedule Found - StartTime: {start_time}, EndTime: {end_time}, CurrentTime: {current_time}")

        # Check if the current time is within the schedule range
        return start_time <= current_time <= end_time

    except Exception as e:
        print(f"Error in is_within_schedule: {e}")
        return False

    finally:
        connection.close()









# -- for roles -- 
def role_required(required_roles):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if 'role' not in session:
                flash('You must be logged in to access this page.', 'danger')
                return redirect(url_for('login'))
            if session['role'] not in required_roles:
                flash('You do not have permission to access this page.', 'danger')
                return redirect(url_for('login'))
            return func(*args, **kwargs)
        return wrapper
    return decorator


@app.route('/check_schedule', methods=['POST'])
def check_schedule():
    try:
        data = request.get_json()
        subject_id = data['subject_id']
        section = data['section']

        # Check if within allowed schedule
        is_allowed = is_within_schedule(subject_id, section)

        return jsonify({"is_allowed": is_allowed})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        SID = request.form['sid']
        Name = request.form['name']

        if not SID or not Name:
            flash("Please enter both SID and Name!", "danger")
            return redirect(url_for('register'))

        if student_exists(SID, Name):
            flash(f"Student data exists with SID '{SID}'!", "danger")
            return redirect(url_for('register'))
        
        # Redirect to the camera page
        return redirect(url_for('register_process', sid=SID, name=Name))

    return render_template('register.html')



@app.route('/register_process/<sid>/<name>', methods=['GET'])
def register_process(sid, name):
    """
    Render the live camera feed page for registering a student.
    """
    return render_template('register_process.html', sid=sid, name=name)


@app.route('/capture_complete/<sid>/<name>', methods=['GET'])
def capture_complete(sid, name):
    """
    After completing image capture, redirect to register page with a success message.
    """
    flash(f"Successfully registered student: {name} (SID: {sid})", "success")
    return redirect(url_for('register'))



@app.route('/video_feed/<sid>/<name>')
def video_feed(sid, name):
    """
    Route to serve live video feed during image capture.
    """
    return Response(capture_images(sid, name), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/registration_complete')
def registration_complete():
    flash("Student successfully registered!", "success")
    return redirect(url_for('register'))





# Global event for recognition completion
recognition_complete = threading.Event()

# Route for subject input page
@app.route('/face_recognition', methods=['GET', 'POST'])
@role_required(['lecturer', 'admin']) 
def face_recognition():
    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    
    # Fetch subjects with sections for the lecturer
    lecturer_id = session.get('ID')
    cursor.execute("""
        SELECT s.SubjectCode, lss.Section
        FROM Lecturer_has_Subject lss
        JOIN Subjects s ON lss.SubjectID = s.SubjectID
        WHERE lss.LecturerID = %s
    """, (lecturer_id,))
    subject_section_pairs = cursor.fetchall()
    connection.close()

    # Organize subjects and sections for rendering
    subjects = []
    for pair in subject_section_pairs:
        subjects.append({
            'SubjectCode': pair['SubjectCode'],
            'Section': pair['Section']
        })

    if request.method == "POST":
        subject_section = request.form.get("subject")
        if not subject_section:
            flash("Please select a subject and section.", "danger")
            return redirect(url_for('face_recognition'))

        # Split the subject and section
        subject_code, section = subject_section.split('|')

        # Redirect with subject and section
        return redirect(url_for('live_feed', subject=subject_code, section=section))

    return render_template("face_recognition.html", subjects=subjects)




@app.route('/student_attendance/<student_id>', methods=['GET'])
def student_attendance(student_id):
    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    try:
        # Fetch the student's name
        cursor.execute("""
            SELECT Name FROM StudentDetails WHERE SID = %s
        """, (student_id,))
        student = cursor.fetchone()
        if not student:
            flash(f"Student with ID {student_id} not found.", "danger")
            return redirect(url_for('lecturer_dashboard'))
    

        cursor.execute("""
            SELECT ar.SubjectID, s.SubjectName, s.SubjectCode, COUNT(*) AS TotalClasses,
                SUM(ar.Attended) AS ClassesAttended
            FROM AttendanceRecords ar
            JOIN Subjects s ON ar.SubjectID = s.SubjectID
            WHERE ar.StudentID = %s
            GROUP BY ar.SubjectID, s.SubjectName, s.SubjectCode;
        """, (student_id,))  

    

        attendance_data = cursor.fetchall()

        # Add calculated percentages
        for record in attendance_data:
            total_classes = record['TotalClasses']
            attended = record['ClassesAttended']
            record['AttendancePercentage'] = round((attended / total_classes) * 100, 2) if total_classes > 0 else 0

        return render_template(
            'student_attendance.html',
            student_name=student['Name'],
            student_id=student_id,
            attendance=attendance_data
        )

    except Exception as e:
        print(f"Error fetching attendance for student {student_id}: {e}")
        flash("An error occurred while fetching student attendance.", "danger")
        return redirect(url_for('lecturer_dashboard'))
    finally:
        connection.close()



# FETCH STUDENTS PERFORMANCE FROM SUBJECT
def fetch_class_attendance(subject_code):
    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    try:
        # Fetch attendance for all students in the class (ignoring section)
        cursor.execute("""
            SELECT 
                ar.StudentID, 
                ar.Name AS StudentName, 
                s.SubjectCode, 
                s.SubjectName, 
                COUNT(*) AS TotalClasses, 
                SUM(ar.Attended) AS ClassesAttended, 
                ROUND(SUM(ar.Attended) / COUNT(*) * 100, 2) AS AttendancePercentage
            FROM AttendanceRecords ar
            JOIN Subjects s ON ar.SubjectID = s.SubjectID
            WHERE s.SubjectCode = %s
            GROUP BY ar.StudentID, ar.Name, s.SubjectCode, s.SubjectName;
        """, (subject_code,))  # Only pass subject_code as parameter

        attendance_data = cursor.fetchall()

        if not attendance_data:
            flash(f"No attendance records found for {subject_code}.", "warning")
            return render_template('class_attendance.html', subject_code=subject_code, attendance=[])

        return render_template(
            'class_attendance.html',
            subject_code=subject_code,
            attendance=attendance_data
        )

    except Exception as e:
        print(f"Error fetching attendance for {subject_code}: {e}")
        flash("Error fetching attendance data.", "danger")
        return render_template('class_attendance.html', subject_code=subject_code, attendance=[])
    finally:
        cursor.close()
        connection.close()


# CHOOSE SUBJECT FOR PERFORMANCE
@app.route('/choose_subject', methods=['GET', 'POST'])
def choose_subject():
    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    try:
        # Fetch all available subjects
        cursor.execute("SELECT SubjectCode, SubjectName FROM Subjects")
        subjects = cursor.fetchall()

        if request.method == 'POST':
            # Get the selected subject code from the form
            selected_subject_code = request.form.get('subject')
            if not selected_subject_code:
                flash("Please select a subject.", "danger")
            else:
                # Redirect to the class attendance page for the selected subject
                return redirect(url_for('class_attendance', subject_code=selected_subject_code))

        return render_template('choose_subject.html', subjects=subjects)

    except Exception as e:
        print(f"Error fetching subjects: {e}")
        flash("Error fetching subjects.", "danger")
        return render_template('choose_subject.html', subjects=[])
    finally:
        cursor.close()
        connection.close()



# Class Attendance
@app.route('/class_attendance/<subject_code>', methods=['GET'])
def class_attendance(subject_code):
    return fetch_class_attendance(subject_code)


# Route for live recognition feed
@app.route('/recognition_feed/<subject>/<section>')
@role_required(['lecturer', 'admin'])
def recognition_feed(subject, section):
    return Response(
        take_attendance(subject, section), 
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )




# Route for the live feed page
@app.route('/live_feed/<subject>/<section>', methods=["GET", "POST"])
@role_required(['lecturer', 'admin'])
def live_feed(subject, section):
    if request.method == "POST":
        flash(f"Attendance for {subject} - Section {section} has been successfully recorded!", "success")
        return redirect(url_for('face_recognition'))
    return render_template('live_feed.html', subject=subject, section=section)



@app.route('/finalize_attendance/<subject>/<section>', methods=['POST'])
def finalize_attendance(subject, section):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(pymysql.cursors.DictCursor)

        # Fetch all registered students for the subject and section
        cursor.execute("""
            SELECT sd.ID, sd.SID, sd.Name, ss.SubjectID, ss.Section
            FROM StudentDetails sd
            INNER JOIN Student_has_Subject ss ON sd.ID = ss.StudentID
            WHERE ss.SubjectID = (SELECT SubjectID FROM Subjects WHERE SubjectCode = %s)
            AND ss.Section = %s
        """, (subject, section))
        registered_students = cursor.fetchall()

        # Ensure the list is not empty
        if not registered_students:
            flash("No students registered for this subject.", "warning")
            return redirect(url_for('face_recognition'))

        # Create a DataFrame for all registered students
        all_students_df = pd.DataFrame(registered_students)
        all_students_df["Date"] = datetime.now().strftime('%Y-%m-%d')
        all_students_df["Time"] = datetime.now().strftime('%H:%M:%S')
        all_students_df["Attended"] = 0  # Default to not attended
        all_students_df["SessionID"] = f"{subject}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        # Handle attendance data if available
        if attendance_records:
            # Remove duplicates based on SID in attendance_records
            unique_attendance_records = {record["SID"]: record for record in attendance_records}.values()
            detected_students_df = pd.DataFrame(unique_attendance_records)
            detected_students_df["Attended"] = 1  # Mark as attended

            # Merge detected attendance with all registered students
            all_students_df = all_students_df.merge(
                detected_students_df[["SID", "Attended"]], 
                on="SID", 
                how="left", 
                suffixes=("", "_detected")
            )
            all_students_df["Attended"] = all_students_df["Attended_detected"].fillna(0).astype(int)
            all_students_df.drop(columns=["Attended_detected"], inplace=True)

        # Remove duplicate rows before inserting into the database
        all_students_df.drop_duplicates(subset=["SID", "Name"], inplace=True)

        # Insert all attendance records into the AttendanceRecords table
        for _, student in all_students_df.iterrows():
            cursor.execute("""
                INSERT INTO AttendanceRecords (StudentID, Name, SubjectID, Section, Date, Time, Attended, SessionID)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                student["SID"], 
                student["Name"], 
                student["SubjectID"], 
                student["Section"], 
                student["Date"], 
                student["Time"], 
                student["Attended"],
                student["SessionID"]
            ))

        connection.commit()
        connection.close()

        flash(f"Attendance for {subject} successfully saved!", "success")
    except Exception as e:
        print(f"Error saving attendance: {e}")
        flash("Error saving attendance.", "danger")
    return redirect(url_for('face_recognition'))





# Route for video feed with face detection
@app.route('/capture_feed/<SID>/<Name>')
def capture_feed(SID, Name):
    return Response(generate_capture_feed(SID, Name), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for add image page
@app.route("/add_images", methods=["GET", "POST"])
def add_image():
    if request.method == "POST":
        SID = request.form['SID']
        Name = request.form['Name']

        if not SID or not Name:
            flash("Please enter both SID and Name!", "danger")
            return redirect(url_for('add_images'))

        if not student_exists(SID, Name):
            flash(f"No student found with SID '{SID}'!", "danger")
            return redirect(url_for('add_image'))

        print(f"Capturing images for {Name} (SID: {SID}).", "info")
        return redirect(url_for('capture', SID=SID, Name=Name))

    return render_template("add_images.html")

# Route to capture images
@app.route("/capture/<SID>/<Name>", methods=["GET", "POST"])
def capture(SID, Name):
    if request.method == "POST":
        output_folder = capture_images_addon(SID, Name)
        if output_folder:
            image_files = [f for f in os.listdir(output_folder) if f.startswith(f"{Name}.{SID}") and f.endswith(".jpg")]
            image_count = len(image_files)
            flash(f"{image_count} images successfully saved for {SID} - {Name}", "success")
        else:
            flash("Error capturing images!", "danger")
        return redirect(url_for('add_image'))

    return render_template("capture.html", SID=SID, Name=Name)



@app.route('/train_images', methods=['GET', 'POST'])
@role_required(['admin'])
def train_images():
    print("train_images function was called!")  # Debugging statement
    if request.method == 'POST':
        try:
                time_module.sleep(2) 
                print("train_images function was called! 222!") 
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

                def get_images_and_labels(path):
                    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
                    face_samples = []
                    ids = []

                    for image_path in image_paths:
                        if image_path == 'TrainingImage/.DS_Store':
                            pass
                        else:
                            try:
                                pil_image = Image.open(image_path).convert('L')  # Grayscale image
                                image_np = np.array(pil_image, 'uint8')
                                Id1 = os.path.split(image_path)[-1].split(".")[1]
                                id = int(re.search(r'\d+', Id1).group())
                                faces = detector.detectMultiScale(image_np)

                                for (x, y, w, h) in faces:
                                    face_samples.append(image_np[y:y + h, x:x + w])
                                    ids.append(id)
                                    print(f"{image_path} is processed")

                            except Exception as e:
                                print(f"Error processing file '{image_path}': {e}")

                        if not face_samples or not ids:
                            print("No valid face samples or IDs were found.")
                            flash("No valid face samples or IDs were found.", "danger")

                    flash(f"Processed {len(face_samples)} face samples.", "success")
                    return face_samples, ids

                path = "TrainingImage"
                faces, ids = get_images_and_labels(path)
                print(f"Files: {len(faces)}")
                recognizer.train(faces, np.array(ids))
                recognizer.write("TrainingImageLabel\Trainner.yml")  # Save model
                
                print('train_img function finish!')
                return redirect(url_for('train_images'))
        except Exception as e:
                flash(f"Error during training: {e}", "danger")

    return render_template('train_img.html')

@app.route('/view_sessions', methods=['GET', 'POST'])
@role_required(['lecturer', 'admin'])
def view_sessions():
    try:
        connection = get_db_connection()
        cursor = connection.cursor(pymysql.cursors.DictCursor)

        if session['role'] == 'admin':
            # Admin can view all sessions
            cursor.execute("""
                SELECT DISTINCT SessionID, SubjectID, Date, Time
                FROM AttendanceRecords
                ORDER BY Date DESC, Time DESC
            """)
        else:
            # Lecturer can view sessions only for their assigned subjects
            user_id = session['ID']
            cursor.execute("""
                SELECT DISTINCT ar.SessionID, ar.SubjectID, ar.Date, ar.Time
                FROM AttendanceRecords ar
                INNER JOIN Lecturer_has_Subject ls ON ar.SubjectID = ls.SubjectID
                WHERE ls.LecturerID = %s
                ORDER BY ar.Date DESC, ar.Time DESC
            """, (user_id,))

        sessions = cursor.fetchall()

        # Format session display for the dropdown
        formatted_sessions = []
        for session_data in sessions:
            session_code = session_data['SessionID'].split('-')[0]  # Extract SubjectCode
            session_date = session_data['Date']  # Raw Date from query
            session_time = session_data['Time']  # Raw Time from query

            # Ensure session_date and session_time are strings for formatting

            session_date = datetime.strptime(str(session_date), '%Y-%m-%d').strftime('%d-%m-%Y')  # Format date
            session_time = datetime.strptime(str(session_time), '%H:%M:%S').strftime('%H:%M')  # Format time

            formatted_sessions.append({
                'SessionID': session_data['SessionID'],
                'Display': f"{session_code} ({session_date} @ {session_time})"
            })


        if request.method == "POST":
            selected_session = request.form.get("session")
            if not selected_session:
                flash("Please select a session.", "warning")
                return redirect(url_for('view_sessions'))

            # Fetch attendance data for the selected session
            cursor.execute("""
                SELECT StudentID, Name, Attended, Section
                FROM AttendanceRecords
                WHERE SessionID = %s
                ORDER BY Name ASC
            """, (selected_session,))
            attendance_data = cursor.fetchall()

            # Calculate attendance statistics
            total_students = len(attendance_data)
            attended_count = sum(record["Attended"] for record in attendance_data)
            absent_count = total_students - attended_count

            connection.close()

            return render_template(
                "view_sessions.html",
                sessions=formatted_sessions, 
                attendance_data=attendance_data,
                selected_session=selected_session,
                total_students=total_students,
                attended_count=attended_count,
                absent_count=absent_count
            )

        connection.close()
        return render_template("view_sessions.html", sessions=formatted_sessions, attendance_data=None)

    except Exception as e:
        print(f"Error viewing sessions: {e}")
        flash("Error retrieving sessions.", "danger")
        return redirect(url_for('index'))





# Route for attendance
@app.route('/attendance', methods=['GET', 'POST'])
@role_required(['lecturer','admin'])
def attendance():
    if request.method == 'POST':
        subject = request.form['subject']
        if not subject:
            flash("Subject name is required.", "danger")
        else:
            # Attendance logic here
            flash("Attendance filled successfully.", "success")
    return render_template('attendance.html')


# ----- LOGIN REGISTER STUFFS ---------
# Route for the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        connection = get_db_connection()
        cursor = connection.cursor(pymysql.cursors.DictCursor)

        cursor.execute("SELECT * FROM Users WHERE Username = %s", (username,))
        user = cursor.fetchone()

        if user and check_password_hash(user['PasswordHash'], password):
            session['ID'] = user['ID']
            session['role'] = user['Role']
            session['username'] = user['Username']  # Add username to session
            if user['Role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif user['Role'] == 'lecturer':
                return redirect(url_for('lecturer_dashboard'))
            else:
                flash("Invalid role specified.", "danger")
                return redirect(url_for('login'))
        else:
            flash("Invalid username or password.", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')




# Route for the register page
@app.route('/register_acc', methods=['GET', 'POST'])
def register_acc():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        name = request.form['name']
        email = request.form['email']

        # Hash the password
        hashed_password = generate_password_hash(password)

        try:
            connection = get_db_connection()
            cursor = connection.cursor()
            
            cursor.execute("""
                INSERT INTO Users (Username, PasswordHash, Role, Name, Email)
                VALUES (%s, %s, %s, %s, %s)
            """, (username, hashed_password, role, name, email))

            connection.commit()
            connection.close()

            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            flash(f'Error during registration: {e}', 'danger')

    return render_template('register_acc.html')

#Route for edit attendance page
@app.route('/edit_attendance/<session_id>', methods=['POST'])
@role_required(['lecturer', 'admin'])
def update_attendance(session_id):
    try:
        # Debugging: Print the raw form data
        print("Request form data:", request.form)

        # Initialize attendance_status as an empty dictionary
        attendance_status = {}

        # Parse nested attendance_status keys manually
        for key, value in request.form.items():
            if key.startswith('attendance_status[') and key.endswith(']'):
                student_id = key[18:-1]  # Extract the student ID from the key
                attendance_status[student_id] = int(value)

        # Debugging: Print the parsed attendance_status
        print("Parsed attendance_status:", attendance_status)

        if not attendance_status:
            flash("No attendance data received!", "danger")
            return redirect(url_for('view_sessions', session=session_id))

        # Update the database
        connection = get_db_connection()
        cursor = connection.cursor()

        for student_id, status in attendance_status.items():
            cursor.execute("""
                UPDATE AttendanceRecords
                SET Attended = %s
                WHERE SessionID = %s AND StudentID = %s
            """, (status, session_id, student_id))

        connection.commit()
        connection.close()

        flash("Attendance updated successfully!", "success")
        return redirect(url_for('view_sessions', session=session_id))

    except Exception as e:
        print(f"Error updating attendance: {e}")
        flash("Error updating attendance.", "danger")
        return redirect(url_for('view_sessions', session=session_id))



# Download Report
@app.route('/download_report/<session_id>', methods=['GET'])
def download_report(session_id):
    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)

    try:
        # Fetch attendance data for the session
        cursor.execute("""
            SELECT ar.StudentID, sd.Name, ar.Section, ar.Attended
            FROM AttendanceRecords ar
            JOIN StudentDetails sd ON ar.StudentID = sd.SID
            WHERE ar.SessionID = %s
        """, (session_id,))
        attendance_data = cursor.fetchall()

        # Prepare the CSV
        def generate_csv():
            # Define the header
            header = ['Student ID', 'Name', 'Section', 'Attendance Status']
            # Write the header
            yield ','.join(header) + '\n'
            # Write the rows
            for record in attendance_data:
                attendance_status = 'Present' if record['Attended'] == 1 else 'Absent'
                row = [
                    record['StudentID'],
                    record['Name'],
                    record['Section'],
                    attendance_status
                ]
                yield ','.join(row) + '\n'

        # Stream the CSV to the client
        response = Response(generate_csv(), mimetype='text/csv')
        response.headers.set(
            "Content-Disposition",
            f"attachment; filename=attendance_report_{session_id}.csv"
        )
        return response

    except Exception as e:
        flash(f"Error generating report: {e}", "danger")
        return redirect(url_for('view_sessions'))  # Redirect back to the session view

    finally:
        cursor.close()
        connection.close()



# Admin dashboard
@app.route('/admin_dashboard')
@role_required(['admin'])
def admin_dashboard():
    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    try:
        # Fetch user information
        query = "SELECT Name FROM Users WHERE ID = %s"
        cursor.execute(query, (session['ID'],))
        user = cursor.fetchone()

        if user:
            session['user_name'] = user['Name']
        else:
            session['user_name'] = 'Unknown'  # Default to "Unknown" if user not found
    finally:
        cursor.close()
        connection.close()

    return render_template('admin_dashboard.html', user_name=session['user_name'])



# Lecturer dashboard
@app.route('/lecturer_dashboard', methods=['GET'])
def lecturer_dashboard():

    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    try:
        # Fetch user information
        query = "SELECT Name FROM Users WHERE ID = %s"
        cursor.execute(query, (session['ID'],))
        user = cursor.fetchone()

        if user:
            session['user_name'] = user['Name']
        else:
            session['user_name'] = 'Unknown'  # Default to "Unknown" if user not found
    finally:
        cursor.close()
        connection.close()

    return render_template('lecturer_dashboard.html', user_name=session['user_name'])



# ASSIGN SUBJECTS TO LECTURER 
@app.route('/assign_subjects', methods=['GET', 'POST'])
def assign_subjects():
    connection = get_db_connection()  # Initialize database connection
    try:
        if request.method == 'POST':
            lecturer_id = request.form.get('lecturer')
            subject_id = request.form.get('subject')
            section = request.form.get('sections') or ''  # Optional field

            try:
                # Insert or update the record
                query = """
                    INSERT INTO Lecturer_has_Subject (LecturerID, SubjectID, Section)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE Section = VALUES(Section)
                """
                cursor = connection.cursor()
                cursor.execute(query, (lecturer_id, subject_id, section))
                connection.commit()
                flash('Subject successfully assigned to lecturer!', 'success')
            except Exception as e:
                connection.rollback()
                flash(f'Error assigning subject: {e}', 'danger')
            finally:
                cursor.close()

            return redirect(url_for('assign_subjects'))
        else:
            # Fetch lecturers from the Users table where Role is 'lecturer'
            cursor = connection.cursor(pymysql.cursors.DictCursor)
            cursor.execute("SELECT ID, Name FROM Users WHERE Role = 'lecturer'")
            lecturers = cursor.fetchall()

            # Fetch subjects from the Subjects table
            cursor.execute("SELECT SubjectID, SubjectCode, SubjectName FROM Subjects")
            subjects = cursor.fetchall()

            cursor.close()
            return render_template('assign_subjects.html', lecturers=lecturers, subjects=subjects)
    finally:
        connection.close()  # Ensure the connection is closed



# edit Profile page
@app.route('/edit_profile', methods=['GET', 'POST'])
@role_required(['lecturer', 'admin'])  # Ensure only lecturers can access
def edit_profile():
    user_id = session.get('ID')  # Get the logged-in user's ID from session
    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    
    # Fetch current user details
    cursor.execute("SELECT Name, Email FROM Users WHERE ID = %s", (user_id,))
    user = cursor.fetchone()

    if request.method == 'POST':
        new_name = request.form.get('name')
        new_email = request.form.get('email')
        new_password = request.form.get('password')
        
        # Update user details
        if new_name and new_email:
            cursor.execute(
                "UPDATE Users SET Name = %s, Email = %s WHERE ID = %s",
                (new_name, new_email, user_id)
            )
            flash("Profile updated successfully!", "success")
        
        # Update password if provided
        if new_password:
            hashed_password = generate_password_hash(new_password)
            cursor.execute(
                "UPDATE Users SET PasswordHash = %s WHERE ID = %s",
                (hashed_password, user_id)
            )
            flash("Password updated successfully!", "success")

        connection.commit()
        connection.close()

        return redirect(url_for('edit_profile'))
    
    connection.close()
    return render_template('edit_profile.html', user=user)

# Assign Student to subject
@app.route('/assign_student_to_subject', methods=['GET', 'POST'])
@role_required(['admin'])  # Only admins can access this page
def assign_student_to_subject():
    connection = get_db_connection()
    try:
        cursor = connection.cursor(pymysql.cursors.DictCursor)

        if request.method == 'POST':
            # Get form data
            student_id = request.form.get('student')
            subject_id = request.form.get('subject')
            section = request.form.get('section')

            # Validate form inputs
            if not student_id or not subject_id or not section:
                flash("Please fill in all fields: student, subject, and section.", "warning")
                return redirect(url_for('assign_student_to_subject'))

            # Insert or update the assignment in the Student_has_Subject table
            try:
                query = """
                    INSERT INTO Student_has_Subject (StudentID, SubjectID, Section)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE Section = VALUES(Section)
                """
                cursor.execute(query, (student_id, subject_id, section))
                connection.commit()
                flash("Student successfully assigned to the subject and section.", "success")
            except Exception as e:
                connection.rollback()
                flash(f"Error assigning student: {e}", "danger")
            finally:
                cursor.close()

            return redirect(url_for('assign_student_to_subject'))

        # Fetch students and subjects for the dropdowns
        cursor.execute("SELECT ID, SID, Name FROM StudentDetails")
        students = cursor.fetchall()

        cursor.execute("SELECT SubjectID, SubjectCode, SubjectName FROM Subjects")
        subjects = cursor.fetchall()

        cursor.close()
        return render_template('assign_student_to_subject.html', students=students, subjects=subjects)

    finally:
        connection.close()


# View assigned students
@app.route('/student_subject', methods=['GET', 'POST'])
@role_required(['admin', 'lecturer'])
def student_subject():
    connection = get_db_connection()
    try:
        cursor = connection.cursor(pymysql.cursors.DictCursor)

        search_query = request.form.get('search_query', '').strip()
        base_query = """
            SELECT DISTINCT
                ss.StudentID,
                sd.Name AS StudentName,
                ss.Section,
                s.SubjectCode,
                s.SubjectName,
                u.Name AS LecturerName
            FROM 
                Student_has_Subject ss
            INNER JOIN 
                StudentDetails sd ON ss.StudentID = sd.ID
            INNER JOIN 
                Subjects s ON ss.SubjectID = s.SubjectID
            INNER JOIN 
                Lecturer_has_Subject ls ON ss.SubjectID = ls.SubjectID
            INNER JOIN 
                Users u ON ls.LecturerID = u.ID
            WHERE 
                u.Role = 'lecturer'
        """

        if search_query:
            base_query += """
                AND (sd.Name LIKE %s OR s.SubjectCode LIKE %s)
            """
            cursor.execute(base_query, (f"%{search_query}%", f"%{search_query}%"))
        else:
            cursor.execute(base_query)

        assigned_students = cursor.fetchall()
        cursor.close()

        return render_template('student_subject.html', assigned_students=assigned_students, search_query=search_query)

    finally:
        connection.close()


# DElete assigned student subject
@app.route('/delete_assigned_student', methods=['POST'])
def delete_assigned_student():
    student_id = request.args.get('student_id')
    subject_id = request.args.get('subject_id')

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        # Delete the assignment from Student_has_Subject table
        cursor.execute("""
            DELETE FROM Student_has_Subject 
            WHERE StudentID = %s AND SubjectID = (SELECT SubjectID FROM Subjects WHERE SubjectCode = %s)
        """, (student_id, subject_id))

        connection.commit()
        flash("Student assignment deleted successfully!", "success")
    except Exception as e:
        print(f"Error deleting assignment: {e}")
        flash("Failed to delete student assignment.", "danger")
    finally:
        cursor.close()
        connection.close()

    return redirect(url_for('student_subject'))




# View Lect and SUbject page
@app.route('/lecturer_subject', methods=['GET', 'POST'])
def lecturer_subject():
    connection = get_db_connection()
    try:
        cursor = connection.cursor(pymysql.cursors.DictCursor)

        # Check for a search query in the POST request
        search_query = request.form.get('search_query', '').strip()

        if search_query:
            # If a search query is provided, filter results
            query = """
                SELECT LS.LecturerID, U.Name AS LecturerName, S.SubjectCode, S.SubjectName, LS.Section
                FROM Lecturer_has_Subject LS
                JOIN Users U ON LS.LecturerID = U.ID
                JOIN Subjects S ON LS.SubjectID = S.SubjectID
                WHERE U.Role = 'lecturer'
                AND (S.SubjectName LIKE %s OR S.SubjectCode LIKE %s)
            """
            cursor.execute(query, (f"%{search_query}%", f"%{search_query}%"))
        else:
            # Default query to fetch all assignments
            query = """
                SELECT LS.LecturerID, U.Name AS LecturerName, S.SubjectCode, S.SubjectName, LS.Section
                FROM Lecturer_has_Subject LS
                JOIN Users U ON LS.LecturerID = U.ID
                JOIN Subjects S ON LS.SubjectID = S.SubjectID
                WHERE U.Role = 'lecturer'
            """
            cursor.execute(query)

        assignments = cursor.fetchall()
        cursor.close()

        return render_template('lecturer_subject.html', assignments=assignments, search_query=search_query)
    finally:
        connection.close()


# Delete button for lectrurrr - subject assignment
@app.route('/delete_assignment', methods=['POST'])
def delete_assignment():
    lecturer_id = request.args.get('lecturer_id')
    subject_id = request.args.get('subject_id')

    connection = get_db_connection()
    cursor = connection.cursor()

    try:
        # Delete the assignment from LecturerSubject table
        cursor.execute("""
            DELETE FROM Lecturer_has_Subject 
            WHERE LecturerID = %s AND SubjectID = (SELECT SubjectID FROM Subjects WHERE SubjectCode = %s)
        """, (lecturer_id, subject_id))

        connection.commit()
        flash("Assignment deleted successfully!", "success")
    except Exception as e:
        print(f"Error deleting assignment: {e}")
        flash("Failed to delete assignment.", "danger")
    finally:
        cursor.close()
        connection.close()

    return redirect(url_for('lecturer_subject'))



# Route to render the insert schedule page
@app.route('/insert_schedule', methods=['GET', 'POST'])
def insert_schedule():
    connection = get_db_connection()
    cursor = connection.cursor(pymysql.cursors.DictCursor)
    
     # Fetch all available SubjectIDs and SubjectCodes from the Subjects table
    cursor.execute("SELECT DISTINCT SubjectID, SubjectName, SubjectCode FROM Subjects WHERE SubjectID IS NOT NULL AND SubjectCode IS NOT NULL")
    subjects = cursor.fetchall()
    

    if request.method == 'POST':
        subject_id = request.form['subject_id']
        section = request.form['section']
        start_time = request.form['start_time']
        end_time = request.form['end_time']
        day_of_week = request.form['day_of_week']

        try:
            # Ensure proper parameterization of the query
            query = """
                INSERT INTO SubjectAttendanceSchedule (SubjectID, Section, StartTime, EndTime, DayOfWeek)
                VALUES (%s, %s, %s, %s, %s)
            """
            print(f"Attempting to insert: SubjectID={subject_id}, Section={section}, StartTime={start_time}, EndTime={end_time}, DayOfWeek={day_of_week}")

            # Format `start_time` and `end_time` to match the database format (HH:MM:SS)
            formatted_start_time = f"{start_time}:00" if len(start_time) == 5 else start_time
            formatted_end_time = f"{end_time}:00" if len(end_time) == 5 else end_time

            # Execute the query
            cursor.execute(query, (subject_id, section, formatted_start_time, formatted_end_time, day_of_week))
            connection.commit()

            flash("Schedule added successfully!", "success")
            return redirect(url_for('insert_schedule'))
        except Exception as e:
            flash(f"Error adding schedule: {e}", "danger")
            return redirect(url_for('insert_schedule'))

        finally:
            connection.close()

   
    connection.close()

    return render_template('insert_schedule.html', subjects=subjects, debug_data=subjects)





# Logout route
@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)



