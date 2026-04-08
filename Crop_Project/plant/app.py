from flask import Flask, render_template, request, redirect, url_for, session
import tensorflow as tf
import numpy as np
import os
import uuid
import json
import cv2
import sqlite3
from flask import send_from_directory
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash
from tensorflow.keras.applications.efficientnet import preprocess_input
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
import difflib
from datetime import datetime, timedelta
import os
from flask import flash
import random


def generate_otp():
    return str(random.randint(100000, 999999))


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# ===============================
# CREATE FLASK APP FIRST
# ===============================
app = Flask(__name__)

# ✅ Secret key MUST be after app creation
app.secret_key = "crop_health_secret_key"
# Default session lifetime (for Remember Me)
app.permanent_session_lifetime = timedelta(days=7)

print(app.url_map)
@app.before_request
def make_session_temporary():
    if "user" in session:
        expiry = session.get("expiry")

        if expiry:
            expiry_time = datetime.fromisoformat(expiry)
            if datetime.utcnow() > expiry_time:
                session.clear()
                return redirect(url_for("login"))


# ================= EMAIL CONFIGURATION =================

app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USE_SSL"] = False
app.config["MAIL_USERNAME"] = "krushimitraai2026@gmail.com"
app.config["MAIL_PASSWORD"] = "plhnimlcrddwsisp"

mail = Mail(app)

serializer = URLSafeTimedSerializer(app.secret_key)

# ================================
# PATHS + CONFIG
# ================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease.keras")
JSON_PATH = os.path.join(BASE_DIR, "models", "plant_disease.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploading_images")

# set upload folder in flask config
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# create upload folder if not exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ================================
# LOAD MODEL
# ================================
print("Loading Model...")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Model Loaded Successfully ✅")

# ================================
# LOAD JSON
# ================================
with open(JSON_PATH) as f:
    plant_disease_list = json.load(f)

# Convert list to dictionary
plant_disease = {item["name"]: item for item in plant_disease_list}

# class names for prediction
class_names = sorted([item["name"] for item in plant_disease_list])

print("JSON Loaded Successfully ✅")

# ==========================================
# DISEASE KNOWLEDGE BASE (Chatbot)
# ==========================================

disease_knowledge = {
    "Tomato - Septoria leaf spot": {
        "treatment": "Apply Mancozeb or Chlorothalonil fungicide every 7-10 days.",
        "organic": "Use Neem oil spray or copper-based fungicide.",
        "prevention": "Avoid overhead watering and remove infected leaves immediately.",
        "duration": "Improvement may be seen within 10-14 days.",
    },
    "Healthy": {
        "general": "Your crop is healthy. Maintain balanced watering and nutrients."
    },
}

print("JSON Loaded Successfully ✅")
print("JSON Loaded Successfully ✅")


# ================================
# LEAF VALIDATION USING OPENCV
# ================================
def is_leaf(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask > 0) / (img.shape[0] * img.shape[1])

    if green_ratio < 0.08:
        return False

    return True


# ================================
# IMAGE PREPROCESSING (IMPORTANT FIX)
# ================================
def preprocess_image(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)

    # 🔥 MATCH TRAINING PREPROCESSING
    img_array = preprocess_input(img_array)

    return img_array


# ================================
# PREDICTION FUNCTION
# ================================
def predict_disease(image_path):

    # Step 1: Leaf validation
    if not is_leaf(image_path):
        return {
            "name": "Invalid Image",
            "cause": "The uploaded image does not appear to be a plant leaf.",
            "cure": "Please upload a clear green plant leaf image.",
            "confidence": 0,
            "fertilizer": "Not Applicable",
            "products": {},
        }

    # Step 2: Preprocess
    img = preprocess_image(image_path)

    # Step 3: Model Prediction
    predictions = model.predict(img)[0]
    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index]) * 100

    # Step 4: Get correct class name
    predicted_class_name = class_names[predicted_index]

    # Step 5: Get JSON info
    disease_info = plant_disease.get(predicted_class_name)

    if not disease_info:
        return {
            "name": "Prediction Error",
            "cause": "Class not found in JSON.",
            "cure": "Check model and JSON mapping.",
            "confidence": 0,
            "fertilizer": "Not Available",
            "products": {},
        }

    return {
        "name": disease_info["name"],
        "cause": disease_info["cause"],
        "cure": disease_info["cure"],
        "confidence": round(confidence, 2),
        "translations": disease_info.get("translations", {}),
        "fertilizer": disease_info.get("fertilizer", "Consult Agronomist"),
        "products": disease_info.get("products", {}),
    }


# For Language Translation
def get_language():
    lang = request.args.get("lang", "en")
    if lang not in ["en", "hi", "mr"]:
        return "en"
    return lang


def init_db():
    conn = sqlite3.connect("users.db", timeout=10, check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute(
        """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    role TEXT NOT NULL,
    reset_token TEXT,
    reset_token_expiry TEXT,
    is_verified INTEGER DEFAULT 0,
    email_token TEXT,
    otp TEXT,
    otp_expiry TEXT
)
"""
    )

    # CONTACT MESSAGES TABLE
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS contact_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        mobile TEXT,
        email TEXT,
        location TEXT,
        crop TEXT,
        problem_type TEXT,
        message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """
    )

    conn.commit()
    conn.close()


# ================================
# ROUTES
# ================================

# ================================
# ROUTES
# ================================

# # Helper function to detect language
# def get_language():
#     lang = request.args.get("lang", "en")
#     if lang not in ["en", "hi", "mr"]:
#         lang = "en"
#     return lang


@app.route("/")
def home():
    lang = get_language()
    return render_template("home.html", lang=lang)


@app.route("/predict")
def predict_page():

    if "user" not in session:
        return redirect(url_for("login"))

    lang = get_language()
    return render_template("predict.html", lang=lang)


@app.route("/upload/", methods=["POST"])
def upload():

    lang = get_language()  # 🔥 Get language first

    if "img" not in request.files:
        return redirect(url_for("predict_page", lang=lang))

    file = request.files["img"]

    if file.filename == "":
        return redirect(url_for("predict_page", lang=lang))

    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(save_path)

    prediction = predict_disease(save_path)
    session["last_disease"] = prediction["name"]

    print("DEBUG session disease:", session["last_disease"])
    print("DEBUG prediction name repr:", repr(prediction["name"]))

    print("DEBUG Disease Name:", prediction["name"])
    # 🔥 Store predicted disease for chatbot context
    session["last_disease"] = prediction["name"]

    return render_template(
        "predict.html",
        result=True,
        imagepath=url_for("uploaded_file", filename=unique_filename),
        prediction=prediction,
        lang=lang,  # 🔥 VERY IMPORTANT
    )


#  For Login
@app.route("/uploading_images/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db", timeout=10, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()

        # ✅ Check credentials
        if user and check_password_hash(user[3], password):

            # 🔐 Block unverified users
            is_verified = (
                user[7] if len(user) > 7 else 0
            )  # Handle missing column gracefully

            if is_verified == 0:
                session.clear()  # Clear any existing session data
                session["verify_email"] = user[2]  # email store
                flash("Please verify your email using OTP", "warning")
                return redirect(url_for("verify_otp"))

            # 🔐 Store session data
            session["user"] = user[1]
            session["role"] = user[4]

            # ✅ Remember Me
            remember = request.form.get("remember")

            if remember:
                expiry_time = datetime.utcnow() + timedelta(days=7)
            else:
                expiry_time = datetime.utcnow() + timedelta(minutes=30)

            session["expiry"] = expiry_time.isoformat()

            # 🔀 Role-based redirect
            if user[4] == "admin":
                return redirect(url_for("admin_dashboard"))
            else:
                return redirect(url_for("home"))

        else:
            return render_template("login.html", error="Invalid Username or Password")

    # GET request
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        role = request.form["role"]

        hashed_password = generate_password_hash(password)

        try:
            otp = str(generate_otp())
            expiry = datetime.now() + timedelta(minutes=5)

            with sqlite3.connect("users.db", timeout=10) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO users (username, email, password, role, otp, otp_expiry, is_verified)
                    VALUES (?, ?, ?, ?, ?, ?, 0)
                """,
                    (username, email, hashed_password, role, otp, expiry.isoformat()),
                )

                conn.commit()

            msg = Message(
                "Your OTP - Crop Health Diagnostic System",
                sender=app.config["MAIL_USERNAME"],
                recipients=[email],
            )

            msg.subject = "🔐 Verify Your Email | 🌿कृषि मित्र AI"

            msg.html = f"""
<div style="font-family: 'Segoe UI', Arial; background:#f4f6f8; padding:30px;">

    <div style="max-width:600px; margin:auto; background:white; border-radius:12px; padding:25px; box-shadow:0 10px 30px rgba(0,0,0,0.1);">

        <h2 style="color:#2e7d32; margin-bottom:10px;">
            🌿कृषि मित्र AI
        </h2>

        <p style="color:#555;">Hello {username},</p>

        <p style="color:#555;">
            Welcome! Please verify your email using the OTP below:
        </p>

        <div style="
            background:linear-gradient(135deg,#e8f5e9,#ffffff);
            padding:18px;
            text-align:center;
            font-size:32px;
            font-weight:bold;
            border-radius:10px;
            margin:20px 0;
            letter-spacing:3px;
            color:#1b5e20;
        ">
            {otp}
        </div>

        <p style="color:#555;">
            ⏳ This OTP is valid for <b>5 minutes</b>.
        </p>

        <p style="color:#777; font-size:13px;">
            If you didn’t request this, please ignore this email.
        </p>

        <hr>

        <p style="font-size:12px; color:#aaa;">
            © 2026 🌿कृषि मित्र AI | 🌱 Crop Health Diagnostic System 🌱
        </p>

    </div>

</div>
"""
            mail.send(msg)

            session["verify_email"] = email
            return redirect(url_for("verify_otp"))

        except sqlite3.IntegrityError:
            return render_template("register.html", error="User already exists")

    return render_template("register.html")

# Otp Verification Route
@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():

    email = session.get("verify_email")

    if not email:
        return render_template(
            "verify_otp.html", error="Session expired. Please register again."
        )

    if request.method == "POST":

        user_otp = request.form["otp"].strip()

        with sqlite3.connect("users.db", timeout=10) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT otp, otp_expiry, is_verified 
                FROM users WHERE email=?
            """,
                (email,),
            )

            user = cursor.fetchone()

            if user:
                db_otp, expiry, is_verified = user

                print("Entered OTP:", user_otp)
                print("DB OTP:", db_otp)

                if is_verified == 1:
                    return render_template(
                        "login.html", success="Already verified. Please login."
                    )

                if not expiry or datetime.now() > datetime.fromisoformat(expiry):
                    return render_template(
                        "verify_otp.html", error="OTP expired. Click resend."
                    )

                if str(user_otp).strip() == str(db_otp).strip():

                    cursor.execute(
                        """
                        UPDATE users 
                        SET is_verified=1, otp=NULL, otp_expiry=NULL 
                        WHERE email=?
                    """,
                        (email,),
                    )

                    conn.commit()

                    session.pop("verify_email", None)

                    return render_template(
                        "verify_otp.html",
                        success="✅ Email verified successfully!",
                        show_login_button=True,
                        email=None,
                    )

        return render_template("verify_otp.html", error="Invalid OTP")

    return render_template("verify_otp.html", email=email)

@app.route("/resend-otp")
def resend_otp():

    # 🔐 Step 1: Get email from session
    email = session.get("verify_email")

    if not email:
        return redirect(url_for("login"))

    # 🔥 Step 2: Generate new OTP
    otp = str(generate_otp())
    expiry = datetime.now() + timedelta(minutes=5)

    # 🔐 Step 3: Update OTP in DB
    with sqlite3.connect("users.db", timeout=10) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE users 
            SET otp=?, otp_expiry=? 
            WHERE email=?
        """,
            (otp, expiry.isoformat(), email),
        )

        conn.commit()

    # 🔥 Step 4: Send email
    msg = Message(
        subject="🔄 New OTP Requested | Krushi Mitra AI",
        sender=app.config["MAIL_USERNAME"],
        recipients=[email],
    )

    msg.subject = "🔄 New OTP Requested | 🌿कृषि मित्र AI"

    msg.html = f"""
<div style="font-family: 'Segoe UI', Arial; background:#f4f6f8; padding:30px;">

    <div style="max-width:600px; margin:auto; background:white; border-radius:12px; padding:25px;">

        <h2 style="color:#2e7d32;">🌿कृषि मित्र AI</h2>

        <p>Hello,</p>

        <p>You requested a new OTP for verification.</p>

        <div style="
            background:#e8f5e9;
            padding:15px;
            text-align:center;
            font-size:28px;
            font-weight:bold;
            border-radius:10px;
            margin:20px 0;
            letter-spacing:2px;
        ">
            {otp}
        </div>

        <p>⏳ Valid for <b>5 minutes</b></p>

        <p style="color:#777; font-size:13px;">
            If this wasn’t you, please ignore this email.
        </p>

        <hr>

        <p style="font-size:12px; color:#aaa;">
            Secure Notification |🌿कृषि मित्र AI 🌱
        </p>

    </div>

</div>
"""
    mail.send(msg)

    # 🔁 Step 5: Redirect back to verify page
    return redirect(url_for("verify_otp"))


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():

    if request.method == "POST":
        email = request.form["email"].strip()

        with sqlite3.connect("users.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE email=?", (email,))
            user = cursor.fetchone()

        if user:
            otp = str(generate_otp())
            expiry = datetime.now() + timedelta(minutes=5)

            with sqlite3.connect("users.db") as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users 
                    SET otp=?, otp_expiry=? 
                    WHERE email=?
                """, (otp, expiry.isoformat(), email))
                conn.commit()

            session["reset_email"] = email

            # ✅ email sending INSIDE if
            msg = Message(
    subject="🔐 Password Reset OTP | 🌿कृषि मित्र AI",
    sender=app.config["MAIL_USERNAME"],
    recipients=[email],
)

            msg.html = f"""
<div style="font-family: 'Segoe UI', Arial; background:#f4f6f8; padding:30px;">

    <div style="max-width:600px; margin:auto; background:white; border-radius:14px; overflow:hidden; box-shadow:0 10px 25px rgba(0,0,0,0.1);">

        <!-- Header -->
        <div style="background: linear-gradient(135deg, #2e7d32, #66bb6a); padding:20px; text-align:center; color:white;">
            <h2 style="margin:0;">🌿कृषि मित्र AI</h2>
            <p style="margin:5px 0 0;">Secure Password Reset</p>
        </div>

        <!-- Body -->
        <div style="padding:25px; text-align:center;">

            <h3 style="color:#333;">🔐 Reset Your Password</h3>

            <p style="color:#555;">
                Use the OTP below to reset your password.
            </p>

            <!-- OTP BOX -->
            <div style="
                background:#e8f5e9;
                padding:18px;
                font-size:30px;
                font-weight:bold;
                border-radius:12px;
                margin:20px auto;
                letter-spacing:3px;
                color:#1b5e20;
                display:inline-block;
                min-width:200px;
            ">
                {otp}
            </div>

            <p style="color:#777;">
                ⏳ Valid for <b>5 minutes</b>
            </p>

            <p style="font-size:13px; color:#999;">
                If you didn’t request this, please ignore this email.
            </p>

        </div>

        <!-- Footer -->
        <div style="background:#f1f1f1; padding:15px; text-align:center; font-size:12px; color:#888;">
            © 2026 🌿कृषि मित्र AI 🌱 | Secure Notification
        </div>

    </div>

</div>
"""

            mail.send(msg)

            return redirect(url_for("verify_reset_otp"))

        return render_template("forgot_password.html", error="Email not found")

    return render_template("forgot_password.html")

@app.route("/resend-reset-otp", endpoint="resend_reset_otp_new")
def resend_reset_otp():

    email = session.get("reset_email")

    if not email:
        return redirect(url_for("forgot_password"))

    otp = str(generate_otp())
    expiry = datetime.now() + timedelta(minutes=5)

    with sqlite3.connect("users.db") as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users 
            SET otp=?, otp_expiry=? 
            WHERE email=?
        """, (otp, expiry.isoformat(), email))
        conn.commit()

    msg = Message(
    subject="🔐 Password Reset OTP | 🌿कृषि मित्र AI",
    sender=app.config["MAIL_USERNAME"],
    recipients=[email],
)

    msg.html = f"""
<div style="font-family: 'Segoe UI', Arial; background:#f4f6f8; padding:30px;">

    <div style="max-width:600px; margin:auto; background:white; border-radius:14px; overflow:hidden; box-shadow:0 10px 25px rgba(0,0,0,0.1);">

        <!-- Header -->
        <div style="background: linear-gradient(135deg, #2e7d32, #66bb6a); padding:20px; text-align:center; color:white;">
            <h2 style="margin:0;">🌿कृषि मित्रAI</h2>
            <p style="margin:5px 0 0;">Secure Password Reset</p>
        </div>

        <!-- Body -->
        <div style="padding:25px; text-align:center;">

            <h3 style="color:#333;">🔐 Reset Your Password</h3>

            <p style="color:#555;">
                Use the OTP below to reset your password.
            </p>

            <!-- OTP BOX -->
            <div style="
                background:#e8f5e9;
                padding:18px;
                font-size:30px;
                font-weight:bold;
                border-radius:12px;
                margin:20px auto;
                letter-spacing:3px;
                color:#1b5e20;
                display:inline-block;
                min-width:200px;
            ">
                {otp}
            </div>

            <p style="color:#777;">
                ⏳ Valid for <b>5 minutes</b>
            </p>

            <p style="font-size:13px; color:#999;">
                If you didn’t request this, please ignore this email.
            </p>

        </div>

        <!-- Footer -->
        <div style="background:#f1f1f1; padding:15px; text-align:center; font-size:12px; color:#888;">
            © 2026 🌿कृषि मित्र AI 🌱 | Secure Notification
        </div>

    </div>

</div>
"""
    mail.send(msg)
    flash("✅ New OTP sent successfully!", "success") 
    return redirect(url_for("verify_reset_otp"))  # 🔥 IMPORTANT

@app.route("/verify-reset-otp", methods=["GET", "POST"])
def verify_reset_otp():

    email = session.get("reset_email")

    if not email:
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        user_otp = request.form["otp"].strip()

        with sqlite3.connect("users.db") as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT otp, otp_expiry FROM users WHERE email=?", (email,))
            user = cursor.fetchone()

        if user:
            db_otp, expiry = user

            if not expiry or datetime.now() > datetime.fromisoformat(expiry):
                return render_template("verify_reset_otp.html", error="OTP expired")

            if str(user_otp).strip() == str(db_otp).strip():
                return redirect(url_for("reset_password_otp"))  # 🔥 IMPORTANT

        return render_template("verify_reset_otp.html", error="Invalid OTP")

    return render_template("verify_reset_otp.html")

@app.route("/reset-password-otp", methods=["GET", "POST"])
def reset_password_otp():

    email = session.get("reset_email")

    if not email:
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        new_password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if new_password != confirm_password:
            return render_template("reset_password.html", error="Passwords do not match")

        hashed_password = generate_password_hash(new_password)

        with sqlite3.connect("users.db") as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE users 
                SET password=?, otp=NULL, otp_expiry=NULL 
                WHERE email=?
            """, (hashed_password, email))
            conn.commit()

        session.pop("reset_email", None)

        return render_template(
            "success.html",
            title="Password Updated Successfully",
            message="Now login with new password",
            redirect_url=url_for("login"),
            button_text="Go to Login"
        )

    return render_template("reset_password.html")

@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):

    try:
        email = serializer.loads(token, salt="password-reset", max_age=1800)
    except:
        return "Reset link is invalid or expired."

    if request.method == "POST":
        new_password = request.form["password"]

        hashed_password = generate_password_hash(new_password)

        with sqlite3.connect("users.db", timeout=10) as conn:
            cursor = conn.cursor()

            cursor.execute(
                "UPDATE users SET password=? WHERE email=?",
                (hashed_password, email),
            )

            conn.commit()

        return render_template(
            "success.html",
            title="Password Updated Successfully",
            message="Your password has been changed.",
            redirect_url=url_for("login"),
            button_text="Go to Login",
        )

    return render_template("reset_password.html")


@app.route("/admin")
def admin_dashboard():

    # 🔐 Must be logged in
    if "user" not in session:
        return redirect(url_for("login"))

    # 🔐 Must be admin
    if session.get("role") != "admin":
        return "Access Denied. Admins only."

    conn = sqlite3.connect("users.db", timeout=10)
    # conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Pagination
    page = request.args.get("page", 1, type=int)
    per_page = 5
    offset = (page - 1) * per_page

    # Search & Filter
    search = request.args.get("search", "")
    role_filter = request.args.get("role", "")
    status_filter = request.args.get("status", "")

    base_query = "SELECT id, username, email, role, is_verified FROM users WHERE 1=1"
    params = []

    if search:
        base_query += " AND (username LIKE ? OR email LIKE ?)"
        params.extend([f"%{search}%", f"%{search}%"])

    if role_filter:
        base_query += " AND role = ?"
        params.append(role_filter)

    if status_filter == "verified":
        base_query += " AND is_verified = 1"
    elif status_filter == "unverified":
        base_query += " AND is_verified = 0"

    # Total count for pagination
    count_query = f"SELECT COUNT(*) FROM ({base_query})"
    cursor.execute(count_query, params)
    total_filtered_users = cursor.fetchone()[0]
    total_pages = (total_filtered_users + per_page - 1) // per_page

    # Apply limit
    final_query = base_query + " LIMIT ? OFFSET ?"
    params.extend([per_page, offset])
    cursor.execute(final_query, params)
    users = cursor.fetchall()

    # Stats (same as before)
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM users WHERE is_verified = 1")
    verified_users = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM users WHERE is_verified = 0")
    unverified_users = cursor.fetchone()[0]

    conn.close()

    return render_template(
        "admin_dashboard.html",
        users=users,
        total_users=total_users,
        verified_users=verified_users,
        unverified_users=unverified_users,
        page=page,
        total_pages=total_pages,
        search=search,
        role_filter=role_filter,
        status_filter=status_filter,
    )


@app.route("/admin/messages")
def admin_messages():

    # Only admin allowed
    if "user" not in session or session.get("role") != "admin":
        return "Access Denied"

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute(
        """
    SELECT id, name, mobile, crop, problem_type, message, created_at
    FROM contact_messages
    ORDER BY created_at DESC
    """
    )

    messages = cursor.fetchall()

    conn.close()

    return render_template("admin_messages.html", messages=messages)


@app.route("/admin-live-search")
def admin_live_search():

    if "user" not in session or session.get("role") != "admin":
        return {"error": "Unauthorized"}, 403

    search = request.args.get("search", "")
    page = request.args.get("page", 1, type=int)

    per_page = 5
    offset = (page - 1) * per_page

    conn = sqlite3.connect("users.db", timeout=10)
    cursor = conn.cursor()

    if search:
        query = """
            SELECT id, username, email, role, is_verified
            FROM users
            WHERE username LIKE ? OR email LIKE ?
            LIMIT ? OFFSET ?
        """
        cursor.execute(query, (f"%{search}%", f"%{search}%", per_page, offset))
    else:
        query = """
            SELECT id, username, email, role, is_verified
            FROM users
            LIMIT ? OFFSET ?
        """
        cursor.execute(query, (per_page, offset))

    users = cursor.fetchall()
    conn.close()

    user_list = []
    for u in users:
        user_list.append(
            {
                "id": u[0],
                "username": u[1],
                "email": u[2],
                "role": u[3],
                "is_verified": u[4],
            }
        )

    return {"users": user_list}


@app.route("/delete_user/<int:user_id>")
def delete_user(user_id):

    if session.get("role") != "admin":
        return "Access Denied"

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()

    return redirect(url_for("admin_dashboard"))


@app.route("/toggle_role/<int:user_id>")
def toggle_role(user_id):

    if session.get("role") != "admin":
        return "Access Denied"

    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("SELECT role FROM users WHERE id=?", (user_id,))
    current_role = cursor.fetchone()[0]

    new_role = "admin" if current_role == "farmer" else "farmer"

    cursor.execute("UPDATE users SET role=? WHERE id=?", (new_role, user_id))
    conn.commit()
    conn.close()

    return redirect(url_for("admin_dashboard"))


@app.route("/chatbot", methods=["POST"])
def chatbot():

    if "last_disease" not in session:
        return {"reply": "Please upload and diagnose a leaf image first."}

    user_message = request.json.get("message", "").lower()
    disease = session["last_disease"]

    disease_info = plant_disease.get(disease)

    if not disease_info:
        return {"reply": f"No information found for: {disease}"}

    # 🔥 Define Intent Dictionary
    intents = {
        "treatment": ["treatment", "cure", "control", "manage", "solution"],
        "cause": ["cause", "reason", "why"],
        "fertilizer": ["fertilizer", "dose", "quantity", "spray", "how much"],
        "prevention": ["prevent", "avoid", "protection"],
        "improve": ["improve", "recover", "growth", "healthy"],
    }

    # 🔍 Fuzzy Intent Detection
    detected_intent = None

    for intent, keywords in intents.items():
        for word in user_message.split():
            match = difflib.get_close_matches(word, keywords, cutoff=0.7)
            if match:
                detected_intent = intent
                break
        if detected_intent:
            break

    # 🎯 Response Logic
    if detected_intent == "treatment":
        reply = disease_info.get("cure", "Treatment information not available.")

    elif detected_intent == "cause":
        reply = disease_info.get("cause", "Cause information not available.")

    elif detected_intent == "fertilizer":
        fertilizer = disease_info.get(
            "fertilizer", "Fertilizer recommendation not available."
        )
        reply = f"Recommended fertilizer: {fertilizer}. Please follow proper dosage instructions."

    elif detected_intent == "prevention":
        reply = (
            "Remove infected leaves and avoid overwatering. Maintain proper spacing."
        )

    elif detected_intent == "improve":
        reply = "Ensure balanced nutrients, sunlight, and regular monitoring."

    else:
        reply = (
            f"This plant has {disease}. "
            "You can ask about treatment, cause, fertilizer dose, or prevention."
        )

    return {"reply": reply}


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():

    if request.method == "POST":

        name = request.form["name"]
        mobile = request.form["mobile"]
        email = request.form["email"]
        location = request.form["location"]
        crop = request.form["crop"]
        problem_type = request.form["problem_type"]
        message = request.form["message"]

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        cursor.execute(
            """
        INSERT INTO contact_messages
        (name,mobile,email,location,crop,problem_type,message)
        VALUES (?,?,?,?,?,?,?)
        """,
            (name, mobile, email, location, crop, problem_type, message),
        )

        conn.commit()
        conn.close()

        return render_template("contact.html", success="Message sent successfully!")

    return render_template("contact.html")


# @app.route("/predict")
# def predict_page():
#     if "user" not in session:
#         return redirect(url_for("login"))
#     return render_template("predict.html")
# ================================
# RUN SERVER
# ================================
if __name__ == "__main__":
    init_db()
    app.run(debug=True)
