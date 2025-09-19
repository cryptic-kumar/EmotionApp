import os
import re
import base64
import io
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for, flash, jsonify
)
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user
)
from flask_bcrypt import Bcrypt
from models import db, User

import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model, model_from_json

# ===== Flask app setup =====
app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["SECRET_KEY"] = "replace-this-with-a-secure-random-key"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# initialize extensions
db.init_app(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ===== Load model & cascade =====
MODEL_H5 = "emotiondetector.h5"
MODEL_JSON = "emotiondetector.json"
CASCADE_FILE = "haarcascade_frontalface_default.xml"

def load_emotion_model():
    if os.path.exists(MODEL_H5):
        try:
            print("[INFO] Loading model from H5...")
            model = load_model(MODEL_H5)
            print("[INFO] Model loaded from H5.")
            return model
        except Exception as e:
            print("[WARN] load_model failed:", e)

    # fallback to JSON + weights
    if os.path.exists(MODEL_JSON) and os.path.exists(MODEL_H5):
        print("[INFO] Loading model architecture from JSON and weights from H5...")
        with open(MODEL_JSON, "r") as f:
            model_json = f.read()
        model = model_from_json(model_json)
        model.load_weights(MODEL_H5)
        print("[INFO] Model rebuilt from JSON + H5 weights.")
        return model

    raise RuntimeError("Model files not found: place emotiondetector.h5 and/or emotiondetector.json in project folder.")

emotion_model = load_emotion_model()
face_cascade = cv2.CascadeClassifier(CASCADE_FILE)
if face_cascade.empty():
    print("[WARN] Haar cascade may not have loaded. Check haarcascade_frontalface_default.xml exists.")

# ===== Utility helpers =====
def parse_base64_image(data_url):
    # Accepts data:image/...;base64,...
    data = re.sub('^data:image/.+;base64,', '', data_url)
    img_bytes = base64.b64decode(data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def predict_on_frame_bgr(img_bgr):
    """
    Input: BGR OpenCV image (color)
    Returns: list of predictions {label, confidence, box}
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    results = []
    for (x, y, w, h) in faces:
        # ensure ROI in bounds
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(gray.shape[1], x + w), min(gray.shape[0], y + h)
        face_gray = gray[y1:y2, x1:x2]
        try:
            face_resized = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)
        except Exception:
            continue
        face_norm = face_resized.astype("float32") / 255.0
        face_input = np.expand_dims(face_norm, axis=0)  # (1,48,48)
        face_input = np.expand_dims(face_input, axis=-1)  # (1,48,48,1)
        preds = emotion_model.predict(face_input)[0]
        idx = int(np.argmax(preds))
        label = CLASS_LABELS[idx] if idx < len(CLASS_LABELS) else str(idx)
        conf = float(np.max(preds))
        results.append({"label": label, "confidence": conf, "box": [int(x1), int(y1), int(x2-x1), int(y2-y1)]})
    return results

def draw_boxes_on_image(img_bgr, preds):
    """
    Draw rectangles+labels on a copy and return encoded base64 data URL.
    """
    im = img_bgr.copy()
    for p in preds:
        x, y, w, h = p["box"]
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 102, 196), 2)
        text = f'{p["label"]} {p["confidence"]*100:.1f}%'
        cv2.putText(im, text, (x + 4, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 102, 196), 2)
    # encode to JPEG and base64
    _, buffer = cv2.imencode('.jpg', im)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{jpg_as_text}"

# ===== Routes =====
@app.route("/")
def root():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

# ---- Auth views ----
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if not username or not email or not password:
            flash("All fields required")
            return redirect(url_for("register"))
        if User.query.filter((User.username == username) | (User.email == email)).first():
            flash("User with same username/email already exists")
            return redirect(url_for("register"))
        pw_hash = bcrypt.generate_password_hash(password).decode("utf-8")
        user = User(username=username, email=email, password=pw_hash)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please login.")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Invalid credentials")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ---- Dashboard with two choices (predict image | real-time monitor) ----
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=current_user)

# ---- Predict from uploaded image (rendered results) ----
@app.route("/predict-image", methods=["GET", "POST"])
@login_required
def predict_image():
    annotated_dataurl = None
    preds = []
    if request.method == "POST":
        f = request.files.get("file")
        if not f:
            flash("No file uploaded")
            return redirect(request.url)
        # read as OpenCV BGR image
        arr = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            flash("Failed to read image")
            return redirect(request.url)
        preds = predict_on_frame_bgr(img)
        annotated_dataurl = draw_boxes_on_image(img, preds)
    return render_template("predict_image.html", preds=preds, image_data=annotated_dataurl)

# ---- Real-time predictor endpoint (used by monitor page) ----
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    """
    Accepts JSON: {image: "data:image/jpeg;base64,..." } where the image is a full frame
    Returns: {"predictions":[{label,confidence,box}, ...]}
    """
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400
    try:
        img = parse_base64_image(data["image"])
        preds = predict_on_frame_bgr(img)
        return jsonify({"predictions": preds})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Monitor page ----
@app.route("/monitor")
@login_required
def monitor():
    # Monitor page uses static JS/CSS; nothing else needed from server
    return render_template("monitor.html")

# ===== Main =====
if __name__ == "__main__":
    # prepare DB
    os.makedirs(app.instance_path, exist_ok=True)
    with app.app_context():
        db.create_all()
    # run on network so devices on LAN can connect
    app.run(host="0.0.0.0", port=5000, debug=True)
