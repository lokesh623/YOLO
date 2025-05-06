from flask import Flask, render_template, request, redirect, send_from_directory
import os
import subprocess
import uuid
import pathlib

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        video = request.files['video']
        if video.filename == '':
            return redirect(request.url)

        unique_id = str(uuid.uuid4())[:8]
        input_filename = f"{unique_id}_{video.filename}"
        output_filename = f"tracked_{input_filename}"
        input_path = os.path.join(UPLOAD_FOLDER, input_filename)
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        video.save(input_path)

        # Normalize path for subprocess
        input_path_norm = pathlib.Path(input_path).as_posix()
        output_path_norm = pathlib.Path(output_path).as_posix()

        # Run detect.py
        result = subprocess.run([
            'python', 'CT3_Mini_Project/scripts/detect.py',
            '--input', input_path_norm,
            '--output', output_path_norm
        ], capture_output=True, text=True)

        print(result.stdout)
        print(result.stderr)

        # Verify output exists
        if os.path.exists(output_path):
            return render_template('index.html', output_video=output_filename)
        else:
            return f"Processing failed. Check logs."

    return render_template('index.html', output_video=None)

@app.route('/outputs/<filename>')
def output_video(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
