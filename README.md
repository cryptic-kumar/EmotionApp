Filename: README.md

# STEP-BY-STEP USAGE (all files are provided above)

1. Put these files in one project folder:

   - app.py
   - requirements.txt
   - emotiondetector.h5
   - emotiondetector.json
   - haarcascade_frontalface_default.xml
   - static/ (folder containing index.html, script.js, style.css)

2. Create a virtualenv (recommended) and install dependencies:
   python -m venv venv
   source venv/bin/activate # Linux / macOS
   venv\Scripts\activate # Windows
   pip install -r requirements.txt

   NOTE: tensorflow install may take time and requires compatible Python. Use CPU or GPU build per your environment.

3. Run the Flask app:
   python app.py

   You should see logs and confirmation that the model loaded. The app serves the frontend at:
   http://127.0.0.1:5000

4. Open the URL in your browser, allow camera access, click Start. The page will periodically send frames to the backend for prediction.

5. Files usage explanation:

   - emotiondetector.h5 / emotiondetector.json: model architecture + weights. app.py will try to load the full h5; if that fails it will reconstruct model from JSON and load weights from H5.
   - haarcascade_frontalface_default.xml: OpenCV Haar cascade for face detection. Place it in the same folder so the backend can load it.

6. Tuning & troubleshooting:

   - If no face is detected: ensure good lighting, move camera closer, or adjust Haar cascade parameters (scaleFactor, minNeighbors).
   - If model fails to load: check file names and tensorflow version compatibility (json indicates keras 2.15.0).
   - To speed up: increase interval (ms) in UI, reduce video resolution, or run model with GPU-enabled TF.

7. Customizing recommendations:

   - Edit RECOMMENDATIONS mapping in static/script.js to change suggested tasks/content per emotion.

8. Deployment:
   - For local testing, app.run(...) is fine.
   - For production expose via a WSGI server (gunicorn) and, optionally, use HTTPS (camera works only on HTTPS or localhost).

That's it — the server will take frames from browser, detect faces, predict emotion and return suggestions based on the label.
