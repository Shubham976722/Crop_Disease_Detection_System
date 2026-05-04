# 🌿 Crop Health Diagnostic System (AI-Powered)

The Crop Health Diagnostic System is an AI-powered web application designed to detect plant diseases from leaf images and provide actionable insights such as cause, cure, fertilizer recommendation, and prevention methods. This system helps farmers, researchers, and agricultural experts make faster and more accurate decisions using deep learning.

This application allows users to upload an image of a plant leaf, after which the system processes the image, validates it, predicts the disease using a trained deep learning model, and displays detailed results. Along with prediction, it also provides chatbot assistance for better understanding and guidance.

---

## 🚀 Features

The system includes multiple powerful features:

- Upload leaf images for disease detection
- AI-based disease prediction with confidence score
- Display of disease cause, cure, and prevention
- Fertilizer recommendation based on disease
- Chatbot for interactive support
- User authentication (login and registration)
- Email verification system
- Forgot password functionality
- Admin dashboard to manage users
- Contact system for user queries

---

## 🧠 AI Model Details

The project uses a deep learning model trained using TensorFlow and Keras.

- Model Type: Convolutional Neural Network (CNN)
- Input Size: 160 x 160 pixels
- Preprocessing: EfficientNet preprocessing
- Output: Disease classification with confidence score

---

## 🏗️ System Architecture

User uploads image → Flask backend receives image → Image preprocessing → AI model prediction → JSON mapping → Result displayed on frontend

---

## 🔄 Workflow

1. User logs into the system
2. Uploads a plant leaf image
3. Image is saved on the server
4. OpenCV checks if the image is a valid leaf
5. Image is resized and preprocessed
6. AI model predicts the disease
7. Prediction is mapped with JSON data
8. Final result is displayed with detailed information

---

## 🧰 Tech Stack

Backend:
- Python
- Flask

AI / Machine Learning:
- TensorFlow
- Keras
- NumPy

Image Processing:
- OpenCV

Frontend:
- HTML
- CSS
- Bootstrap

Database:
- SQLite

Security:
- Werkzeug (password hashing)
- itsdangerous (token system)

Email Service:
- Flask-Mail (Gmail SMTP)

---

## ⚙️ Installation & Setup

1. Clone the repository:

git clone https://github.com/your-username/Crop_Health_Diagnostic_System.git
cd Crop_Health_Diagnostic_System

2. Create virtual environment:

python -m venv venv
venv\Scripts\activate   (Windows)

3. Install dependencies:

pip install -r requirements.txt

4. Run the application:

python app.py

5. Open browser:

http://127.0.0.1:5000

---

## 📸 How to Use

1. Register or login into the system
2. Upload or capture using camera of a plant leaf image
3. Click on detect
4. View the prediction results
5. Use chatbot for additional help

---

## ⚠️ Important Notes

- Ensure the model file exists in:
  models/plant_disease.keras

- Ensure JSON file exists in:
  models/plant_disease.json

- Internet is required for email verification

---

## 🔐 Security Features

- Password hashing for secure login
- Email verification using token system
- Session management for user control

---

## 🚧 Future Improvements

- Mobile application integration
- More crop disease datasets
- Cloud deployment
- Advanced AI chatbot

---

## 👨‍💻 Author

Narayan Kishor Adhude  
Final Year Computer Science & Enginerring Student  

---

## ⭐ Support

If you like this project, consider giving it a star on GitHub.
