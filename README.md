# Butterfly Classification Web App
# NOTE WEBSITE MAY NOT WORK TO SEE WEBSITE LOCALLY CLONE REPO AND RUN "flaskr --app flask run"
https://butterfly-classifier.onrender.com/
This is a web application for classifying butterfly species using a machine learning model. The app allows users to upload images of butterflies and receive predictions on the species.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Contributing](#contributing)
- [License](#license)

## Features
- User-friendly interface for uploading butterfly images.
- Real-time classification of butterfly species using a trained machine learning model.
- Support for multiple butterfly species.

## Technologies Used
- **Flask**: A micro web framework for Python.
- **Keras**: A high-level neural networks API for building and training models.
- **TensorFlow**: The backend engine for Keras.
- **NumPy**: A library for numerical computations in Python.
- **OpenCV**: A library for image processing.
- **HTML/CSS**: For frontend design.
- **Gunicorn**: A Python WSGI HTTP Server for UNIX.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/butterfly-classification.git
Navigate to the project directory:
bash
Copy code
cd butterfly-classification
Create a virtual environment and activate it:
bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
Install the required packages:
bash
Copy code
pip install -r requirements.txt
Usage
Start the Flask application:
bash
Copy code
python -m flask run
Open your browser and go to http://127.0.0.1:5000.
Upload a butterfly image to receive a classification