from flask import Flask, request, render_template_string, redirect, url_for
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
import librosa

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'ogg'}

# Load the trained model
model_path = r"C:\Users\Akash R H\OneDrive\Desktop\MINI PROJECT\MP\model.pkl"
model = pickle.load(open(model_path, 'rb'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>Parkinson Disease Detection</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-KK94CHFLLe+nY2dmCWGMq91rCGa5gtU4mk92HdvYe+M/SXH301p5ILy+dN9+nJOZ" crossorigin="anonymous">
      <style>
        body {
          font-family: Arial, sans-serif;
          color: #333;
          background-image: url('/static/background.jpeg');
          background-size: cover;
          background-repeat: no-repeat;
          background-position: center center;
        }
        .container {
          margin-top: 50px;
          padding: 30px;
          background-color: rgba(255, 255, 255, 0.9);
          border-radius: 8px;
          box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        h1 {
          color: #007bff;
          font-size: 2.5rem;
          margin-bottom: 20px;
        }
        .btn-primary {
          background-color: #007bff;
          border: none;
        }
        .btn-primary:hover {
          background-color: #0056b3;
        }
        .form-control {
          border: 2px solid #007bff;
          border-radius: 5px;
        }
        .form-control:focus {
          box-shadow: none;
          border-color: #0056b3;
        }
        .alert-message {
          margin-top: 20px;
          padding: 15px;
          border-radius: 5px;
        }
        .alert-danger {
          background-color: #f8d7da;
          color: #721c24;
        }
        .alert-success {
          background-color: #d4edda;
          color: #155724;
        }
        .footer {
          margin-top: 30px;
          padding: 20px;
          background-color: #007bff;
          color: #fff;
          text-align: center;
          border-radius: 8px;
        }
        .footer a {
          color: #fff;
          text-decoration: none;
          font-weight: bold;
        }
        .footer a:hover {
          text-decoration: underline;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>Parkinson Disease Detection</h1>
        <p>Parkinson's disease (PD) is a movement disorder of the nervous system that worsens over time...</p>

        <form action="/predict" method="POST" enctype="multipart/form-data">
          <div class="mb-3">
            <label for="inputData" class="form-label"><i class="fas fa-dna"></i> Upload Voice Signal</label>
            <input type="file" class="form-control" name="file" id="inputData" accept=".wav, .ogg">
            <div id="inputDataHelp" class="form-text">Upload a .wav or .ogg file for Parkinson's disease prediction.</div>
          </div>
          <button type="submit" class="btn btn-primary"><i class="fas fa-paper-plane"></i> Submit</button>
        </form>

        {% if message %}
          <div class="alert-message {% if message_type == 'danger' %}alert-danger{% elif message_type == 'success' %}alert-success{% endif %}">
            <p>{{ message }}</p>
          </div>
        {% endif %}
      </div>
      <div class="footer">
        <p>&copy; 2024 Parkinson Disease Detection. All rights reserved. | <a href="#">Privacy Policy</a> | <a href="#">Terms of Service</a></p>
      </div>
      <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/js/bootstrap.bundle.min.js" integrity="sha384-ENjdO4Dr2bkBIFxQpeoTz1HIcje39Wm4jDKdf19U8gI4ddQ3GYNS7NTKfAdVQSZe" crossorigin="anonymous"></script>
    </body>
    </html>
    '''
    return render_template_string(html)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('index', message="No file part", message_type='danger'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index', message="No selected file", message_type='danger'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Audio processing
        y, sr = librosa.load(file_path, sr=None)
        features = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        
        if prediction == 1:
            output = "This person has Parkinson's disease"
            message_type = 'danger'
        else:
            output = "This person does not have Parkinson's disease"
            message_type = 'success'
        
        return redirect(url_for('index', message=output, message_type=message_type))
    else:
        return redirect(url_for('index', message="Invalid file format", message_type='danger'))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
