"""
AI Attendance System - Complete Streamlit Application
Created by: AI Assistant
Date: October 8, 2025

Features:
- Student Registration with Photo Upload/Capture
- Real-time Face Recognition Attendance
- Excel Report Generation for Absentees
- Professional UI with Custom Styling
- Data Persistence using Pickle/JSON

Required Libraries:
pip install streamlit face_recognition opencv-python pandas xlsxwriter pillow numpy
"""

import streamlit as st
import face_recognition
import pandas as pd
import numpy as np
import os
import io
import pickle
from datetime import datetime, date
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="🎓 AI Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling
def load_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #4CAF50;
        --secondary-color: #2196F3;
        --accent-color: #FF9800;
        --success-color: #4CAF50;
        --warning-color: #FF9800;
        --error-color: #F44336;
        --text-color: #333333;
        --bg-color: #f8f9fa;
    }
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Card styling */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid var(--primary-color);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: bold;
    }
    
    .metric-card p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    </style>
    """, unsafe_allow_html=True)

# Data Management Functions
class AttendanceData:
    def __init__(self):
        self.data_dir = "attendance_data"
        self.students_file = os.path.join(self.data_dir, "students.pkl")
        self.attendance_file = os.path.join(self.data_dir, "attendance.pkl")
        self.images_dir = os.path.join(self.data_dir, "student_images")
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.students = self.load_students()
        self.attendance_records = self.load_attendance()
    
    def load_students(self):
        if os.path.exists(self.students_file):
            try:
                with open(self.students_file, 'rb') as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                return {}
        return {}
    
    def save_students(self):
        with open(self.students_file, 'wb') as f:
            pickle.dump(self.students, f)
    
    def load_attendance(self):
        if os.path.exists(self.attendance_file):
            try:
                with open(self.attendance_file, 'rb') as f:
                    return pickle.load(f)
            except (pickle.UnpicklingError, EOFError):
                return []
        return []
    
    def save_attendance(self):
        with open(self.attendance_file, 'wb') as f:
            pickle.dump(self.attendance_records, f)
    
    # --- FIX APPLIED HERE: Changed to take image_array instead of path ---
    def generate_face_encoding(self, image_array):
        """Generates face encoding from a NumPy image array."""
        try:
            # face_recognition works directly with the NumPy array
            encodings = face_recognition.face_encodings(image_array)
            if encodings:
                return encodings[0]
        except Exception as e:
            st.error(f"Error processing face: {str(e)}")
        return None
    
    # --- FIX APPLIED HERE: Handles PIL to NumPy conversion and encoding generation ---
    def register_student(self, student_id, name, class_name, email, image):
        image_path = os.path.join(self.images_dir, f"{student_id}.jpg")
        
        # 1. Ensure image is RGB (critical for face_recognition)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 2. Convert to NumPy array for face processing
        image_array = np.array(image)
        
        # 3. Generate encoding directly from the array
        face_encoding = self.generate_face_encoding(image_array)
        
        if face_encoding is not None:
            # 4. Save the image (for display/reload later)
            image.save(image_path)
            
            self.students[student_id] = {
                'name': name, 'class': class_name, 'email': email,
                'image_path': image_path, 'face_encoding': face_encoding,
                'registration_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.save_students()
            return True
        return False
    
    def mark_attendance(self, image):
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_array = np.array(image)
            face_encodings = face_recognition.face_encodings(image_array)
            marked_students = []
            
            for face_encoding in face_encodings:
                known_encodings = [data['face_encoding'] for data in self.students.values()]
                if not known_encodings:
                    return []
                    
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    student_id = list(self.students.keys())[best_match_index]
                    student_data = self.students[student_id]
                    today = date.today().strftime("%Y-%m-%d")
                    already_marked = any(
                        r['student_id'] == student_id and r['date'] == today for r in self.attendance_records
                    )
                    
                    if not already_marked:
                        self.attendance_records.append({
                            'student_id': student_id, 'name': student_data['name'],
                            'class': student_data['class'], 'date': today,
                            'time': datetime.now().strftime("%H:%M:%S"), 'status': 'Present'
                        })
                        marked_students.append(student_data['name'])

            if marked_students:
                self.save_attendance()
                return list(set(marked_students))
        except Exception as e:
            st.error(f"Error in face recognition: {str(e)}")
        return []

    def get_attendance_stats(self):
        today = date.today().strftime("%Y-%m-%d")
        present_today_ids = {r['student_id'] for r in self.attendance_records if r['date'] == today}
        total_students = len(self.students)
        present_count = len(present_today_ids)
        return {
            'total_students': total_students,
            'present_today': present_count,
            'absent_today': total_students - present_count,
            'attendance_rate': (present_count / total_students * 100) if total_students > 0 else 0
        }

@st.cache_resource
def get_data_manager():
    return AttendanceData()

data_manager = get_data_manager()

def main():
    load_css()
    st.markdown("""
    <div class="main-header">
        <h1>🎓 AI Attendance System</h1>
        <p>Advanced Face Recognition Based Attendance Management</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/667eea/white?text=AI+Attendance", caption="Smart Attendance Solution")
        st.markdown("### 📊 Quick Stats")
        stats = data_manager.get_attendance_stats()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Students", stats['total_students'])
        with col2:
            st.metric("Present Today", stats['present_today'])
        st.metric("Attendance Rate", f"{stats['attendance_rate']:.1f}%")
        st.markdown("---")
        page = st.selectbox(
            "🧭 Navigation",
            ["📝 Student Registration", "📸 Mark Attendance", "📊 Attendance Reports", "⚙️ System Settings"],
            index=0
        )
    
    if page == "📝 Student Registration":
        student_registration_page()
    elif page == "📸 Mark Attendance":
        mark_attendance_page()
    elif page == "📊 Attendance Reports":
        attendance_reports_page()
    elif page == "⚙️ System Settings":
        system_settings_page()

def student_registration_page():
    st.header("📝 Student Registration")
    
    camera_image = None
    uploaded_image = None

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("Student Information")
        
        with st.form("student_form", clear_on_submit=True):
            student_id = st.text_input("Student ID *", placeholder="e.g., STU001")
            name = st.text_input("Full Name *", placeholder="e.g., John Doe")
            class_name = st.selectbox("Class *", 
                ["Select Class", "Class 9A", "Class 9B", "Class 10A", "Class 10B", "Class 11A", "Class 11B", "Class 12A", "Class 12B"])
            email = st.text_input("Email", placeholder="student@school.edu")
            
            st.subheader("Photo Capture")
            photo_option = st.radio("Choose photo method:", ["📷 Take Photo", "📁 Upload Photo"], key="photo_mode")
            
            if photo_option == "📷 Take Photo":
                camera_image = st.camera_input("Take a clear photo")
            else:
                uploaded_image = st.file_uploader("Upload Photo", type=['jpg', 'jpeg', 'png'])
            
            submitted = st.form_submit_button("🚀 Register Student", use_container_width=True)

            if submitted:
                image_buffer = camera_image or uploaded_image
                if not all([student_id, name, class_name != "Select Class"]):
                    st.error("❌ Please fill all required fields!")
                elif student_id in data_manager.students:
                    st.error("❌ Student ID already exists!")
                elif not image_buffer:
                    st.error("❌ Please provide a photo!")
                else:
                    try:
                        # Image.open() reads the uploaded file buffer
                        image = Image.open(image_buffer) 
                        with st.spinner("🔄 Processing registration..."):
                            # Pass the PIL image object to the manager
                            success = data_manager.register_student(
                                student_id.strip(), name.strip(), class_name, email.strip(), image
                            )
                        
                        if success:
                            st.success(f"✅ {name} registered successfully!")
                            st.balloons()
                        else:
                            st.error("❌ Failed to detect a face in the image. Please try with a clearer photo.")
                    except Exception as e:
                        st.error(f"❌ Registration failed: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("📋 Registration Guidelines")
        st.markdown("""
        *For best results:*
        - ✅ Use good lighting
        - ✅ Face should be clearly visible
        - ✅ Look directly at camera
        - ✅ Remove glasses if possible
        - ✅ Neutral expression
        - ❌ Avoid shadows on face
        - ❌ Don't use blurry images
        """)
        
        st.info("ℹ️ The form clears on successful submission. A preview will not be shown live.")
        st.markdown('</div>', unsafe_allow_html=True)

def mark_attendance_page():
    st.header("📸 Mark Attendance")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("📷 Live Attendance Capture")
        st.markdown("""
        *Instructions:*
        1. Click 'Take Photo' button below.
        2. Position face(s) clearly in the camera frame.
        3. The system will automatically recognize and mark attendance.
        """)
        camera_image = st.camera_input("📸 Capture for Attendance", label_visibility="collapsed")
        
        if camera_image:
            with st.spinner("🔍 Recognizing faces..."):
                image = Image.open(camera_image)
                marked_students = data_manager.mark_attendance(image)
            
            if marked_students:
                st.success(f"✅ Attendance marked for: {', '.join(marked_students)}")
                st.balloons()
            else:
                st.warning("⚠️ No new registered students were recognized, or they were already marked present.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("📊 Today's Attendance")
        today = date.today().strftime("%Y-%m-%d")
        today_attendance = [r for r in data_manager.attendance_records if r['date'] == today]
        
        if today_attendance:
            df = pd.DataFrame(today_attendance)
            st.dataframe(df[['name', 'class', 'time']], use_container_width=True)
        else:
            st.info("📝 No attendance marked today yet.")
        st.markdown('</div>', unsafe_allow_html=True)

def attendance_reports_page():
    st.header("📊 Attendance Reports")
    col1, col2 = st.columns([2, 1])
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("📅 Report Filters")
        selected_date = st.date_input("Select Date", value=date.today())
        classes = ["All Classes"] + sorted(list(set(s['class'] for s in data_manager.students.values())))
        selected_class = st.selectbox("Filter by Class", classes)
        report_type = st.selectbox("Report Type", ["Absentees Only", "All Students Status"])
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        date_str = selected_date.strftime("%Y-%m-%d")
        present_students = {r['student_id'] for r in data_manager.attendance_records if r['date'] == date_str}
        
        report_data = []
        for student_id, student_data in data_manager.students.items():
            if selected_class == "All Classes" or student_data['class'] == selected_class:
                status = 'Present' if student_id in present_students else 'Absent'
                report_data.append({
                    'Student ID': student_id, 'Name': student_data['name'],
                    'Class': student_data['class'], 'Status': status
                })
        
        if not report_data:
            st.warning("No students found for the selected filter.")
        else:
            df = pd.DataFrame(report_data)
            if report_type == "Absentees Only":
                df = df[df['Status'] == 'Absent']
                st.subheader(f"❌ Absentees for {selected_date.strftime('%B %d, %Y')}")
            else:
                st.subheader(f"👥 All Students Status for {selected_date.strftime('%B %d, %Y')}")

            if df.empty:
                st.success("🎉 No records to display (e.g., no absentees).")
            else:
                st.dataframe(df, use_container_width=True)
                excel_buffer = create_excel_report(df, f"Report_{date_str}")
                st.download_button(
                    label="📥 Download Excel Report", data=excel_buffer,
                    file_name=f"Report_{date_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

def create_excel_report(df, sheet_name):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        header_format = workbook.add_format({
            'bold': True, 'text_wrap': True, 'valign': 'top',
            'fg_color': '#4CAF50', 'font_color': 'white', 'border': 1
        })
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
            max_len = max(df[value].astype(str).str.len().max(), len(value)) + 2
            worksheet.set_column(col_num, col_num, max_len)
    return buffer.getvalue()

def system_settings_page():
    st.header("⚙️ System Settings")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("📊 System Statistics")
        stats = data_manager.get_attendance_stats()
        st.metric("Total Registered Students", stats['total_students'])
        st.metric("Total Attendance Records", len(data_manager.attendance_records))
        st.markdown("---")
        st.subheader("🗂 Data Management")
        
        if st.button("🧹 Clear All Attendance Records", use_container_width=True):
            data_manager.attendance_records = []
            data_manager.save_attendance()
            st.success("✅ Attendance records cleared!")
            st.rerun()
        
        if st.button("⚠️ Reset All Data", use_container_width=True, type="primary"):
            if "confirm_reset" not in st.session_state:
                st.session_state.confirm_reset = True
            st.rerun()

        if "confirm_reset" in st.session_state and st.session_state.confirm_reset:
            st.warning("This is irreversible. Are you sure you want to delete all students and records?")
            c1, c2 = st.columns(2)
            if c1.button("YES, I AM SURE", use_container_width=True):
                data_manager.students = {}
                data_manager.attendance_records = []
                data_manager.save_students()
                data_manager.save_attendance()
                # Also delete image files
                try:
                    for f in os.listdir(data_manager.images_dir):
                        os.remove(os.path.join(data_manager.images_dir, f))
                except FileNotFoundError:
                    # Directory might not exist or be empty, ignore
                    pass 
                del st.session_state.confirm_reset
                st.success("✅ All data has been reset!")
                st.rerun()
            if c2.button("CANCEL", use_container_width=True):
                del st.session_state.confirm_reset
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.subheader("👥 Registered Students")
        if data_manager.students:
            students_df = pd.DataFrame([
                {'ID': student_id, 'Name': data['name'], 'Class': data['class']}
                for student_id, data in data_manager.students.items()
            ])
            st.dataframe(students_df, use_container_width=True)
        else:
            st.info("📝 No students registered yet.")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()