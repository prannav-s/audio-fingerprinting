from flask import Flask, render_template, request, jsonify
import os
from fingerprint import add_song_to_db, identify_song

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    print(f"File saved to {file_path}")

    result = identify_song(file_path)
    return jsonify(result)

@app.route('/add_song', methods=['POST'])
def add_song():
    if 'file' not in request.files or not all(k in request.form for k in ('title', 'artist', 'album')):
        return jsonify({"error": "Missing data"}), 400
    file = request.files['file']
    title = request.form['title']
    artist = request.form['artist']
    album = request.form['album']

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    print(f"File saved: {file_path}")

    add_song_to_db(file_path, title, artist, album)
    return jsonify({"message": "Song added successfully"})


if __name__ == '__main__':
    app.run(debug=True)
