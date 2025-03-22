import os
import re
import cv2
import base64
import numpy as np
import tempfile
import traceback
from collections import Counter

from flask import Flask, request, jsonify
from flask_cors import CORS

from deepface import DeepFace
import whisper

app = Flask(__name__)
CORS(app)

# ---------------------------
# 1) Whisper Model Setup
# ---------------------------
whisper_model = whisper.load_model("base")

# We define some filler words to gauge speech "quality"
FILLER_WORDS = {"um", "uh", "like", "you", "know", "er", "ah", "so", "well", "actually"}

def compute_speech_metrics(transcription_result):
    """
    Given a Whisper transcription result, compute:
      - Words Per Minute (WPM)
      - Filler word usage & rate
    """
    segments = transcription_result.get("segments", [])
    if not segments:
        return {
            "wpm": 0.0,
            "filler_rate": 0.0,
            "filler_count": 0,
            "filler_words_used": {}
        }

    total_speaking_time = segments[-1]["end"] - segments[0]["start"]
    if total_speaking_time <= 0:
        return {
            "wpm": 0.0,
            "filler_rate": 0.0,
            "filler_count": 0,
            "filler_words_used": {}
        }

    # Combine the text from all segments
    full_text = " ".join(s["text"] for s in segments)
    words = re.findall(r"\w+", full_text.lower())
    total_words = len(words)
    total_minutes = total_speaking_time / 60.0

    wpm = total_words / total_minutes if total_minutes > 0 else 0.0

    filler_counter = Counter(w for w in words if w in FILLER_WORDS)
    filler_count = sum(filler_counter.values())
    filler_rate = filler_count / total_words if total_words > 0 else 0.0

    return {
        "wpm": wpm,
        "filler_rate": filler_rate,
        "filler_count": filler_count,
        "filler_words_used": dict(filler_counter)
    }

def transcribe_audio(audio_file_path):
    """
    Use the Whisper model to transcribe the audio and compute metrics.
    """
    result = whisper_model.transcribe(audio_file_path, language=None)
    transcript = result.get("text", "")
    detected_lang = result.get("language", "unknown")
    metrics = compute_speech_metrics(result)
    return detected_lang.upper(), transcript, metrics

@app.route("/processAudio", methods=["POST"])
def process_audio():
    """
    Endpoint for audio recording. 
    Expects a file in form data with key "audio".
    Returns transcript, filler words, WPM, etc.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    tmp_path = None

    try:
        # 1) Save file to a temporary path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        # 2) Transcribe and compute speech metrics
        lang, transcript, metrics = transcribe_audio(tmp_path)

        return jsonify({
            "language": lang,
            "transcript": transcript,
            "speechRateWPM": round(metrics["wpm"], 2),
            "fillerRate": round(metrics["filler_rate"], 3),
            "fillerCount": metrics["filler_count"],
            "fillerWordsUsed": metrics["filler_words_used"]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ---------------------------
# 2) DeepFace Frame Analysis
# ---------------------------
@app.route('/analyzeFrame', methods=['POST'])
def analyze_frame():
    """
    Endpoint for emotion detection from base64 frames.
    Expects JSON { "image": "data:image/jpeg;base64,..." }
    Returns bounding box + emotion + processed base64 image
    """
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400

    try:
        # 1) Extract base64 from "image"
        encoded_image = data['image'].split(',')[1]
        np_arr = np.frombuffer(base64.b64decode(encoded_image), np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # 2) Validate
        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # 3) Analyze with DeepFace
        analysis = DeepFace.analyze(
            img,
            actions=['emotion'],
            enforce_detection=False
        )

        # 4) Single or multiple faces
        if isinstance(analysis, list):
            if not analysis:
                return jsonify({'error': 'No face detected'}), 400
            face_data = analysis[0]
        else:
            face_data = analysis

        dominant_emotion = face_data.get('dominant_emotion', 'unknown')
        region = face_data.get('region', {})

        # 5) Draw bounding box
        if region:
            x, y = region.get('x', 0), region.get('y', 0)
            w, h = region.get('w', 0), region.get('h', 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(
                img,
                dominant_emotion,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2
            )

        # 6) Convert processed image back to base64
        _, buffer = cv2.imencode('.jpg', img)
        processed_base64 = base64.b64encode(buffer).decode('utf-8')
        processed_image = f"data:image/jpeg;base64,{processed_base64}"

        return jsonify({
            'emotion': dominant_emotion,
            'image': processed_image
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
