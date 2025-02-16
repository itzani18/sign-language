# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import pickle  # For loading your model
# from PIL import Image  # To process image
# import numpy as np  # For converting image to array
# import traceback

# app = Flask(__name__)
# CORS(app)
# img_aux = []
# # Load the model
# with open("model.p", "rb") as model_file:
#     model = pickle.load(model_file)  # Ensure this matches how you saved the model
# print("Model loaded")

# @app.route("/detect", methods=["POST"])
# def detect():
#     try:
#         if "image" not in request.files:
#             print("point 1")
#             return jsonify({"error": "No image uploaded"}), 400

#         image = request.files["image"]
#         print("point 2")

#         # Open the image using PIL and preprocess it
#         img = Image.open(image)
#         print("point 3")
#         img = img.resize((64, 64))
#         print("point 4")  # Resize to the expected input size of your model
#         img_array = np.array(img.convert("L"))
#         print("point 5")  # Convert to grayscale if required
#         img_array = img_array.reshape(1, 64, 64, 1) 
#         print("point 6") # Reshape as needed for your model
#         img_array = img_array / 255.0
#         print("point 7")  # Normalize the image if necessary

#         # Make a prediction
#         prediction = model.predict([np.asarray(img_array)])
#         print("point 8")
#         sign = np.argmax(prediction)
#         print("point 9")  # Get the class with the highest probability

#         return jsonify({"sign": str(sign)})  # Convert the result to a string and send it back

#     except Exception as e:
#         print("Error:", e)
#         traceback.print_exc()  # Print the full error traceback
#         return jsonify({"error": "Internal server error"}), 500

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow CORS for all routes

# Label dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',26:'Hello',27:'I love you',28:'Yes',29:'No'}

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files['image'].read()
    np_image = np.frombuffer(image, np.uint8)
    frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Process the frame with MediaPipe and model
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                data_aux.append(x_[i] - min(x_))
                data_aux.append(y_[i] - min(y_))

        if len(data_aux) == model.n_features_in_:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]
            return jsonify({"sign": predicted_character})

    return jsonify({"sign": "No sign detected"})

if __name__ == '__main__':
    app.run(debug=True)
