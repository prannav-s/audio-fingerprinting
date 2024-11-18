import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
from database import db_connect
import psycopg2

def extract_fingerprint(file_path):
    try:
        # Read the audio file
        fs, audio_data = wavfile.read(file_path)
        # Compute STFT
        _, _, Zxx = stft(audio_data, fs=fs, nperseg=1024)
        # Calculate the magnitude
        magnitude = np.abs(Zxx)
        # Generate fingerprint
        fingerprint = np.mean(magnitude, axis=1)
        return fingerprint
    except Exception as e:
        print(f"Error extracting fingerprint: {e}")
        return None

def load_fingerprints():
    conn = db_connect()
    fingerprints = []
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, fingerprint FROM songs")
            rows = cursor.fetchall()
            for row in rows:
                song_id = row[0]
                fingerprint = np.frombuffer(row[1], dtype=np.float32)
                fingerprints.append((song_id, fingerprint))
        except Exception as error:
            print(f"Error loading fingerprints: {error}")
        finally:
            cursor.close()
            conn.close()
    return fingerprints

def correlate_fingerprints(sample_fingerprint, db_fingerprints):
    best_match = None
    max_correlation = 0
    for song_id, db_fingerprint in db_fingerprints:
        min_length = min(len(sample_fingerprint), len(db_fingerprint))
        sample_fingerprint = sample_fingerprint[:min_length]
        db_fingerprint = db_fingerprint[:min_length]
        correlation = np.corrcoef(sample_fingerprint, db_fingerprint)[0, 1]
        if correlation > max_correlation:
            max_correlation = correlation
            best_match = song_id
    return best_match, max_correlation

def identify_song(file_path):
    # Step 1: Extract fingerprint from uploaded file
    sample_fingerprint = extract_fingerprint(file_path)
    if sample_fingerprint is None:
        return {"error": "Error extracting fingerprint from the uploaded file."}

    # Step 2: Load fingerprints from the database
    db_fingerprints = load_fingerprints()
    if not db_fingerprints:
        return {"error": "No fingerprints found in the database."}

    # Step 3: Correlate the fingerprints
    best_match, correlation_score = correlate_fingerprints(sample_fingerprint, db_fingerprints)

    # Step 4: Fetch song metadata if a match is found
    if best_match:
        conn = db_connect()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT title, artist, album FROM songs WHERE id = %s", (best_match,))
                song = cursor.fetchone()
                if song:
                    return {
                        "song_id": best_match,
                        "title": song[0],
                        "artist": song[1],
                        "album": song[2],
                        "correlation_score": correlation_score
                    }
            except Exception as error:
                print(f"Error fetching song metadata: {error}")
            finally:
                cursor.close()
                conn.close()

    return {"error": "No matching song found."}


def add_song_to_db(file_path, title, artist, album):
    print(f"Processing file: {file_path}")
    print(f"Metadata: Title={title}, Artist={artist}, Album={album}")
    
    fingerprint = extract_fingerprint(file_path)
    if not fingerprint:
        print("Error: Fingerprint extraction failed.")
        return
    
    print(f"Fingerprint extracted: {len(fingerprint)} bytes")

    conn = db_connect()
    if not conn:
        print("Error: Database connection failed.")
        return

    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO songs (title, artist, album, fingerprint)
            VALUES (%s, %s, %s, %s)
            """,
            (title, artist, album, psycopg2.Binary(fingerprint))
        )
        conn.commit()
        print(f"Song '{title}' added to database successfully.")
    except Exception as error:
        print(f"Error adding song to database: {error}")
    finally:
        cursor.close()
        conn.close()



