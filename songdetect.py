import os
from flask import Flask, render_template, request
import librosa
import numpy as np
from scipy.signal import correlate

# Path to the folder containing song directories and chunks
REFERENCE_CHUNK_FOLDER = r"C:\convolutionalnn\DeepAudioClassification\output_chunks_folder"

# Create a mapping between song names and their chunk paths
song_chunk_mapping = {}
for song_dir in os.listdir(REFERENCE_CHUNK_FOLDER):
    song_path = os.path.join(REFERENCE_CHUNK_FOLDER, song_dir)
    if os.path.isdir(song_path):  # Check if it's a directory
        chunks = [
            os.path.join(song_path, f) 
            for f in os.listdir(song_path) if f.endswith('.wav')
        ]
        song_chunk_mapping[song_dir] = chunks

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

# Function to match full audio to chunks and predict the song
def match_full_audio_to_chunks(full_audio_path, song_chunk_mapping, chunk_duration=3, sr=16000):
    """
    Match a full-length audio file against chunks 30–40 of each song and predict the song.
    """
    audio, _ = librosa.load(full_audio_path, sr=sr)
    chunk_size = int(chunk_duration * sr)  # Convert duration to samples
    num_chunks = len(audio) // chunk_size
    song_scores = {song: 0 for song in song_chunk_mapping.keys()}

    print(f"Analyzing {num_chunks} chunks from the uploaded file...")
    
    for i in range(num_chunks):
        # Extract chunk from the full-length audio
        start = i * chunk_size
        end = start + chunk_size
        audio_chunk = audio[start:end]
        if len(audio_chunk) < chunk_size:  # Skip incomplete chunks
            continue

        # Extract features for the current chunk
        chunk_features, _ = extract_features(audio_chunk, sr=sr, is_path=False)

        # Match this chunk against chunks 30–40 of all songs
        for song, chunks in song_chunk_mapping.items():
            for ref_path in chunks[30:50]:  # Process chunks 30–40
                ref_features, _ = extract_features(ref_path, sr=sr)
                score = correlate_features(chunk_features, ref_features)

                # Debugging: Print scores for this chunk
                print(f"Chunk {i}: Comparing with {ref_path}, Score: {score}")

                # Aggregate scores
                song_scores[song] += score

    # Debugging: Print aggregated scores
    print("Aggregated Song Scores:")
    for song, score in song_scores.items():
        print(f"{song}: {score}")

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

    # Match the full-length file against the song directories
    best_match, song_scores = match_full_audio_to_chunks(file_path, song_chunk_mapping)

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
