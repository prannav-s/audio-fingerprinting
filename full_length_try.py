import os
from flask import Flask, render_template, request
import librosa
import numpy as np
from scipy.signal import correlate

# Path to the folder containing full-length song files
REFERENCE_SONG_FOLDER = r"C:\convolutionalnn\DeepAudioClassification\output_wav_folder"

# Create a mapping between song names and their file paths
song_file_mapping = {
    os.path.splitext(song_file)[0]: os.path.join(REFERENCE_SONG_FOLDER, song_file)
    for song_file in os.listdir(REFERENCE_SONG_FOLDER)
    if song_file.endswith('.wav')
}

# Function to extract audio features
def extract_features(audio, sr=16000, n_mfcc=13, is_path=True):
    """
    Extract MFCCs and Chroma features from an audio file or array.
    - audio: File path or audio array.
    - sr: Sampling rate.
    - n_mfcc: Number of MFCCs to extract.
    - is_path: True if 'audio' is a file path; False if 'audio' is an array.
    """
    if is_path:
        audio, _ = librosa.load(audio, sr=sr)  # Load from path
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    return mfcc, chroma

# Function to calculate similarity
def correlate_features(chunk_features, ref_features):
    """
    Compute the cross-correlation between chunk features and reference features.
    """
    chunk_flat = chunk_features.flatten()
    ref_flat = ref_features.flatten()
    return np.correlate(chunk_flat, ref_flat, mode='valid').max()

# Updated function to match uploaded audio to full-length songs
def match_full_audio_to_songs(full_audio_path, song_file_mapping, sr=16000):
    """
    Match a full-length audio file against full-length songs and predict the song.
    """
    # Load the uploaded audio file
    uploaded_audio, _ = librosa.load(full_audio_path, sr=sr)

    # Prepare features for the uploaded file
    uploaded_features, _ = extract_features(uploaded_audio, sr=sr, is_path=False)

    song_scores = {}

    # Compare the uploaded audio to each reference song
    for song_name, song_path in song_file_mapping.items():
        # Load the reference song
        ref_audio, _ = librosa.load(song_path, sr=sr)
        
        # Extract features for the reference song
        ref_features, _ = extract_features(ref_audio, sr=sr, is_path=False)

        # Compute similarity score
        score = correlate_features(uploaded_features, ref_features)

        # Debugging: Print score for each comparison
        print(f"Comparing uploaded audio with {song_name}, Score: {score}")

        # Store the score
        song_scores[song_name] = score

    # Identify the song with the highest score
    best_match = max(song_scores, key=song_scores.get)

    return best_match, song_scores

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Match the full-length file against the full-length songs
    best_match, song_scores = match_full_audio_to_songs(file_path, song_file_mapping)

    if best_match:
        match_results = {
            "predicted_song": best_match,
            "match_scores": song_scores
        }
        return render_template("result.html", results=match_results)
    else:
        return render_template("result.html", results={"predicted_song": "No matches found", "match_scores": {}})

if __name__ == "__main__":
    app.run(debug=True)
