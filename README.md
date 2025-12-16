# 🤖 AI Attendance System

An **AI-powered Attendance Management System** that automates attendance tracking using computer vision and machine learning. This project leverages face recognition to identify individuals and mark attendance accurately in real time.

---

## 📌 Features

* 🎯 Face recognition–based attendance
* 📷 Real-time camera capture
* 🧠 AI/ML-powered identification
* 🗂 Automatic attendance logging
* 📊 Attendance report generation (CSV/Excel)
* 🔐 Secure and reliable system
* 🖥 User-friendly interface

---

## 🛠 Tech Stack

* **Programming Language:** Python
* **Libraries & Frameworks:**

  * OpenCV
  * NumPy
  * Face Recognition / dlib
  * TensorFlow / Keras (optional)
* **Database:** CSV / SQLite / MySQL
* **Frontend (optional):** HTML, CSS, JavaScript
* **IDE:** VS Code / PyCharm

---

## ⚙️ System Architecture

1. Capture image/video through camera
2. Detect faces using OpenCV
3. Extract facial features
4. Match with stored dataset
5. Mark attendance with timestamp
6. Store records in database/file

---

## 🚀 Installation

1. **Clone the repository**

   ```bash
   [git clone https://github.com/your-username/ai-attendance-system.git
   cd ai-attendance-system](https://github.com/Abhhisheksingh/AI-Based-Attendance-System)
   ```

2. **Create virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

1. Add face images to the dataset folder
2. Train the model (if required)

   ```bash
   python train_model.py
   ```
3. Run the attendance system

   ```bash
   python main.py
   ```
4. Attendance will be saved automatically in the attendance folder

---

## 📂 Project Structure

```
ai-attendance-system/
│── dataset/
│── attendance/
│── models/
│── app.py
│── README.md
```

---

## 📈 Output

* Real-time face detection
* Automatic attendance marking
* Date & time stamped records

---

## 🔮 Future Enhancements

* Cloud database integration
* Mobile application support
* Mask detection support
* Live dashboard and analytics
* Multi-camera support

---



---

## 🙋‍♂️ Author

* **ASbhishek_singh**
* GitHub: [[your-username](https://github.com/your-username)](https://github.com/Abhhisheksingh)

---

⭐ If you like this project, don't forget to star the repository!
