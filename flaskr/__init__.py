import os
import logging
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)

# List of butterfly species
species_names = [
    "Danaus plexippus",
    "Heliconius charitonius",
    "Heliconius erato",
    "Junonia coenia",
    "Lycaena phlaeas",
    "Nymphalis antiopa",
    "Papilio cresphontes",
    "Pieris rapae",
    "Vanessa atalanta",
    "Vanessa cardui"
]

# Function to preprocess the image for the model
def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(220, 220))  # Resize image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image array
    return img_array

# Function to create the Flask app
def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite'),
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)

    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Update model path to be relative to the instance folder
    model_path = os.path.join(app.instance_path, 'butterfly_classifier.h5')

    # Check if the model exists before loading it
    if os.path.exists(model_path):
        global model
        model = load_model(model_path)
        logging.info("Model loaded successfully.")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # Ensure the uploads folder exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        # Get the file from the POST request
        file = request.files['file']
        if file:
            file_path = os.path.join('uploads', secure_filename(file.filename))
            # Save the file to the uploads folder
            file.save(file_path)

            logging.info(f"File saved: {file_path}")

            # Preprocess the image and make a prediction
            img = preprocess_image(file_path)

            # Make prediction with error handling
            try:
                predicted_class = model.predict(img)
                # Determine the predicted class
                if len(predicted_class) > 0:
                    predicted_index = np.argmax(predicted_class, axis=-1)
                    species = species_names[predicted_index[0]] if predicted_index[0] < len(species_names) else "Unknown species"
                else:
                    species = "No prediction made"
            except Exception as e:
                logging.error(f"Error during prediction: {e}")
                species = "Error in prediction"

            # Pass the species name to the results page
            return render_template('result.html', species=species)
        else:
            logging.warning("No file was uploaded.")
            return render_template('result.html', species="No file uploaded")

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000)
