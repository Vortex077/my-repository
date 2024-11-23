from flask import Flask, render_template, request, redirect,url_for
import os
import json
import keras
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.layers import Layer
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.inception_resnet_v2 import preprocess_input
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model("Dermno_RenseNet_02.keras")

# Load class labels
with open('class_indices.json', 'r') as f:
    class_labels = json.load(f)

# Define a function to make predictions
def prediction(image_path):
    try:
        # Load and preprocess the image
        img = load_img(image_path, target_size=(256, 256))
        i = img_to_array(img)
        im = preprocess_input(i)
        input_image = np.expand_dims(im, axis=0)

        # Get predictions
        prediction_scores = model.predict(input_image)
        predicted_class = np.argmax(prediction_scores)
        probability_of_predicted_class = prediction_scores[0][predicted_class] * 100

        # Map predicted class index to label
        predicted_label = class_labels[str(predicted_class)]

        # Display the result (for debugging)
        print(f"Predicted Disease: {predicted_label}")
        print(f"Probability: {probability_of_predicted_class:.2f}%")

        # Return the result as a dictionary
        return {
            "class": predicted_label,
            "confidence": f"{probability_of_predicted_class:.2f}%"
        }

    except FileNotFoundError:
        print(f"File not found: {image_path}")
    except OSError as e:
        print(f"Invalid path: {e}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return {"class": "Error", "confidence": "0.00%"}


@app.route('/')
def index():
    return render_template('home.html')  # Replace 'your_html_filename.html' with your actual HTML file name

@app.route('/diagnose')
def diagnose():
    return render_template('diagnose.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/list')
def list():
    return render_template('list.html')

@app.route('/actinic')
def actinic():
    return render_template('actinic.html')

@app.route('/atopic')
def atopic():
    return render_template('Atopic.html')

@app.route('/benign')
def benign():
    return render_template('Benign.html')

@app.route('/dermat')
def dermat():
    return render_template('Dermat.html')

@app.route('/nevus')
def nevus():
    return render_template('Nevus.html')

@app.route('/melanoma')
def melanoma():
    return render_template('Melanoma.html')

@app.route('/squamous')
def squamous():
    return render_template('squamous.html')

@app.route('/tinea')
def tinea():
    return render_template('Tinea.html')

@app.route('/lesions')
def lesions():
    return render_template('Lesions.html')

@app.route('/submit', methods=['POST'])
def submit():
    if 'my_image' not in request.files:
        return "No file part", 400

    file = request.files['my_image']

    if file.filename == '':
        return "No selected file", 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)  # Save uploaded image to static/uploads

        # Use the prediction function to get results
        result1 = prediction(file_path)

        return render_template(
            'diagnose.html',
            result1=result1,
            image_path=file_path.replace('\\', '/')  # Fix Windows path for HTML rendering
        )


if __name__ == '__main__':
    app.run(debug=True)
