import face_recognition
import base64
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

txts_folder = "txt"
jpgs_folder = "jpg"

@app.route('/upload_txt', methods=['POST'])
def upload_txt():
    try:
        txt_file = request.data
        jpg = base64.b64decode(txt_file)

        # Save binary data as a JPG file
        i = len(os.listdir(jpgs_folder))
        with open(f"{jpgs_folder}/{i}.jpg", "wb") as file:
            file.write(jpg)

        return jsonify({"message": "TXT file converted to JPG"})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/detect_faces', methods=['GET'])
def detect_faces():
    frames = [frame for frame in os.listdir(jpgs_folder) if frame.endswith(".jpg")]
    known_faces = [face for face in os.listdir("Images") if face.endswith(".jpg")]

    results = []

    for frame in frames:
        try:
            frame_load = face_recognition.load_image_file(f"{jpgs_folder}/{frame}")
            face_locations = face_recognition.face_locations(frame_load)
            frame_encodings = face_recognition.face_encodings(frame_load, face_locations)

            for face_o_f in frame_encodings:
                is_unknown = True

                for face in known_faces:
                    loaded_jpg = face_recognition.load_image_file(f"Images/{face}")
                    loaded_encoding = face_recognition.face_encodings(loaded_jpg)

                    result = face_recognition.compare_faces(loaded_encoding, face_o_f)

                    if all(result):
                        is_unknown = False
                        results.append({"frame": frame, "face": face})
                        break

                if is_unknown:
                    results.append({"frame": frame, "face": "Unknown"})

        except:
            continue

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)