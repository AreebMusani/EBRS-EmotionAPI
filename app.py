from flask import Flask, request, jsonify
import cv2
from deepface import DeepFace
import numpy as np
from flask_cors import CORS
import dlib
from imutils import face_utils
import os

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Load pre-trained emotion detection model
model = DeepFace.build_model("Emotion")

@app.route('/')
def index():
    return "Hello from Flask!"

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'succes': 'working'}), 200

#generate try catch for handling errors in python


# @app.route('/analyze_emotion', methods=['POST'])
@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    try:
        if 'image' not in request.files:
            return jsonify({'message': 'No image provided'}), 400
        
        # Read image file from request
        image = request.files['image'].read()
        if not image:
            return jsonify({'message': 'Empty image file'}), 400
        
        # Convert image data to numpy array
        nparr = np.frombuffer(image, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'message': 'Could not decode image'}), 400
        
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use dlib for face detection
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray, 1)
        
        if len(faces) == 0:
            return jsonify({'message': 'No faces detected'}), 400
        
        emotions = []
        for face in faces:
            (x, y, w, h) = face_utils.rect_to_bb(face)
            face_img = img[y:y+h, x:x+w]
            
            # Predict emotion
            emotion_result = DeepFace.analyze(img_path=face_img, actions=['emotion', 'age'], enforce_detection=False)
            if emotion_result:
                emotion = emotion_result[0]
                emotions.append({
                    'bounding_box': {'x': x, 'y': y, 'w': w, 'h': h},
                    'dominant_emotion': emotion.get('dominant_emotion', 'Unknown'),
                    'age': emotion.get('age', 'Unknown')
                })
            else:
                emotions.append({'message': 'Unable to analyze emotion'})
        
        return jsonify({'emotions': emotions}), 200

    except Exception as e:
        print('error occurred:', str(e))
        return jsonify({'message': str(e)}), 500

# @app.route('/detect_emotion', methods=['POST'])
# def detect_emotion():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image file found in the request'}), 400

#     image = request.files['image']
#     image_path = os.path.join('temp_image.jpg')
#     image.save(image_path)

#     try:
#         # Analyze the image using DeepFace
#         analysis = DeepFace.analyze(img_path=image_path, actions=['emotion', 'age'], enforce_detection=False)
#         os.remove(image_path)  # Clean up the saved image file
#         return jsonify({'emotions': analysis, 'age': analysis[0]['age']}), 200
#     except Exception as e:
#         print("e", e)
#         return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


# ngrok http --domain=oarfish-obliging-rooster.ngrok-free.app 5000