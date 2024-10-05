This is a web application for classifying butterfly species using a machine learning model. The app allows users to upload images of butterflies and receive predictions on the species.


Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/butterfly-classification.git
Navigate to the project directory:

bash
Copy code
cd butterfly-classification
Create a virtual environment and activate it:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
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

Upload a butterfly image to receive a classification.