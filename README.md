<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Butterfly Classification Web App</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
            padding: 20px;
            background-color: #f4f4f4;
            color: #333;
        }

        h1 {
            color: #2c3e50;
        }

        h2 {
            color: #2980b9;
        }

        h3 {
            color: #27ae60;
        }

        ul {
            list-style-type: disc;
            margin-left: 20px;
        }

        pre {
            background-color: #ecf0f1;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }

        code {
            font-family: monospace;
            background-color: #e8e8e8;
            padding: 2px 4px;
            border-radius: 4px;
        }

        a {
            color: #2980b9;
        }
    </style>
</head>

<body>

    <h1>Butterfly Classification Web App</h1>
    <p>This is a web application for classifying butterfly species using a machine learning model. The app allows users to upload images of butterflies and receive predictions on the species.</p>

    <h2>Table of Contents</h2>
    <ul>
        <li><a href="#features">Features</a></li>
        <li><a href="#technologies-used">Technologies Used</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#usage">Usage</a></li>
        <li><a href="#model-training">Model Training</a></li>
        <li><a href="#contributing">Contributing</a></li>
        <li><a href="#license">License</a></li>
    </ul>

    <h2 id="features">Features</h2>
    <ul>
        <li>User-friendly interface for uploading butterfly images.</li>
        <li>Real-time classification of butterfly species using a trained machine learning model.</li>
        <li>Support for multiple butterfly species.</li>
    </ul>

    <h2 id="technologies-used">Technologies Used</h2>
    <ul>
        <li><strong>Flask</strong>: A micro web framework for Python.</li>
        <li><strong>Keras</strong>: A high-level neural networks API for building and training models.</li>
        <li><strong>TensorFlow</strong>: The backend engine for Keras.</li>
        <li><strong>NumPy</strong>: A library for numerical computations in Python.</li>
        <li><strong>OpenCV</strong>: A library for image processing.</li>
        <li><strong>HTML/CSS</strong>: For frontend design.</li>
        <li><strong>Gunicorn</strong>: A Python WSGI HTTP Server for UNIX.</li>
    </ul>

    <h2 id="installation">Installation</h2>
    <ol>
        <li>Clone the repository:</li>
        <pre><code>git clone https://github.com/yourusername/butterfly-classification.git</code></pre>
        
        <li>Navigate to the project directory:</li>
        <pre><code>cd butterfly-classification</code></pre>

        <li>Create a virtual environment and activate it:</li>
        <pre><code>python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate</code></pre>

        <li>Install the required packages:</li>
        <pre><code>pip install -r requirements.txt</code></pre>
    </ol>

    <h2 id="usage">Usage</h2>
    <ol>
        <li>Start the Flask application:</li>
        <pre><code>python -m flask run</code></pre>
        
        <li>Open your browser and go to <a href="http://127.0.0.1:5000">http://127.0.0.1:5000</a>.</li>
        
        <li>Upload a butterfly image to receive a classification.</li>
    </ol>

    <h2 id="model-training">Model Training</h2>
    <p>The machine learning model is trained using a dataset of butterfly images. The model architecture includes convolutional layers for feature extraction and a dense output layer for classification.</p>
    <p>To train the model:</p>
    <ol>
        <li>Prepare your dataset by placing images in the <code>leedsbutterfly/images</code> directory.</li>
        <li>Run the training script to create and save the model as <code>butterfly_classifier.h5</code>.</li>
    </ol>

    <h2 id="contributing">Contributing</h2>
    <p>Contributions are welcome! Please feel free to submit a pull request or open an issue for any improvements or bugs you find.</p>

    <h2 id="license">License</h2>
    <p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>

</body>