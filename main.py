from flask import Flask, request, render_template, jsonify
import os
from compute_optical_flow import compute_optical_flow
from motion_consistency import check_motion_consistency

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Max file size 100MB

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')  # Your HTML file should be named index.html

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']

    # If no file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # If file is allowed, save it and process
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Step 1: Compute optical flow
        motion_data = compute_optical_flow(filename, save_heatmap=True)
        if motion_data is None or len(motion_data) == 0:
            return jsonify({'error': 'Optical flow computation failed'}), 400

        # Step 2: Check motion consistency
        mean_magnitude, median_magnitude, motion_variance, motion_score = check_motion_consistency(motion_data)

        # Adaptive detection based on motion score
        if motion_score > 5.0:
            result = "✅ REAL VIDEO DETECTED!"
        elif motion_score < 3.0:
            result = "🚨 FAKE VIDEO DETECTED! 🚨"
        elif median_magnitude < 0.5 and motion_variance > 1.5:
            result = "🚨 FAKE VIDEO DETECTED! 🚨 (Inconsistent motion)"
        else:
            result = "⚠️ UNCERTAIN DETECTION! Check motion heatmaps."

        return jsonify({'result': result})

    return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    # Create the upload folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    app.run(debug=True)
