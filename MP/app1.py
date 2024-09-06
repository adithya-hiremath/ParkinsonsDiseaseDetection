import numpy as np
from flask import Flask, request, render_template_string
import pickle

app = Flask(__name__)

model = pickle.load(open(r"C:\Users\Akash R H\OneDrive\Desktop\MINI PROJECT\MP\model.pkl", 'rb'))

@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>PARKINSON DISEASE DETECTION SYSTEM</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-KK94CHFLLe+nY2dmCWGMq91rCGa5gtU4mk92HdvYe+M/SXH301p5ILy+dN9+nJOZ" crossorigin="anonymous">
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css" integrity="sha512-x07CHscI7rknBwt9t57x09z9p6W27g1Fdh8R2YwZHpV4ST/nM3Zg2kB09KrKJ1Zyb1R91B6xWo/3xgqCEEBGbA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
      <style>
        body {
          font-family: Arial, sans-serif;
          color: #333;
          background-image: url('/static/Images/background.jpeg');
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
        p {
          line-height: 1.6;
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
          border-color: #f5c6cb;
        }
        .alert-success {
          background-color: #d4edda;
          color: #155724;
          border-color: #c3e6cb;
        }
        .alert-proper {
          background-color: #eced8a;
          color: #495204;
          border-color: #c3e6cb;
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
      <div class="background-wrapper">
        <div class="container">
          <h1>PARKINSON DISEASE DETECTION MODEL</h1>
          <p>Parkinson's disease (PD) is a movement disorder of the nervous system that worsens over time. As nerve cells (neurons) in parts of the brain weaken, are damaged, or die, people may notice problems with movement, tremor, stiffness in the limbs or trunk, or impaired balance. As symptoms progress, people may have difficulty walking, talking, or completing other simple tasks. Not everyone with one or more of these symptoms has PD, as the symptoms appear in other diseases as well.</p>

          <form action="/predict" method="POST">
            <div class="mb-3">
              <label for="inputData" class="form-label"><i class="fas fa-dna"></i> Input Gait Data</label>
              <input type="text" class="form-control" name="text" id="inputData" aria-describedby="inputDataHelp">
              <div id="inputDataHelp" class="form-text">Enter the data to be analyzed for Parkinson's disease prediction.</div>
            </div>
            <button type="submit" class="btn btn-primary"><i class="fas fa-paper-plane"></i> Submit</button>
          </form>

          {% if message %}
            {% if message_type == 'danger' %}
              <div class="alert-message alert-danger">
                <p>{{ message }}</p>
              </div>
            {% elif message_type == 'success' %}
              <div class="alert-message alert-success">
                <p>{{ message }}</p>
              </div>
            {% else %}
              <div class="alert-message alert-proper">
                <p>{{ message }}</p>
              </div>
            {% endif %}
          {% endif %}
        </div>
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
    input_text = request.form['text']
    input_text_sp = input_text.split(',')
    np_data = np.asarray(input_text_sp, dtype=np.float32)
    prediction = model.predict(np_data.reshape(1, -1))

    if prediction == 1:
        output = "This person has a Parkinson's disease"
        message_type = 'danger'
    elif prediction == 0:
        output = "This person has no Parkinson's disease"
        message_type = 'success'
    else:
        output = "Enter in ASCII CSV Format"
        message_type = 'proper'

    html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <title>PARKINSON DISEASE DETECTION SYSTEM</title>
      <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-KK94CHFLLe+nY2dmCWGMq91rCGa5gtU4mk92HdvYe+M/SXH301p5ILy+dN9+nJOZ" crossorigin="anonymous">
      <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css" integrity="sha512-x07CHscI7rknBwt9t57x09z9p6W27g1Fdh8R2YwZHpV4ST/nM3Zg2kB09KrKJ1Zyb1R91B6xWo/3xgqCEEBGbA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
      <style>
        body {
          font-family: Arial, sans-serif;
          color: #333;
          background-image: url('/static/Images/background.jpeg');
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
        p {
          line-height: 1.6;
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
          border-color: #f5c6cb;
        }
        .alert-success {
          background-color: #d4edda;
          color: #155724;
          border-color: #c3e6cb;
        }
        .alert-proper {
          background-color: #eced8a;
          color: #495204;
          border-color: #c3e6cb;
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
      <div class="background-wrapper">
        <div class="container">
          <h1>Parkinson Disease Prediction Model</h1>
          <p>Parkinson's disease (PD) is a movement disorder of the nervous system that worsens over time. As nerve cells (neurons) in parts of the brain weaken, are damaged, or die, people may notice problems with movement, tremor, stiffness in the limbs or trunk, or impaired balance. As symptoms progress, people may have difficulty walking, talking, or completing other simple tasks. Not everyone with one or more of these symptoms has PD, as the symptoms appear in other diseases as well.</p>

          <form action="/predict" method="POST">
            <div class="mb-3">
              <label for="inputData" class="form-label"><i class="fas fa-dna"></i> Input Data</label>
              <input type="text" class="form-control" name="text" id="inputData" aria-describedby="inputDataHelp">
              <div id="inputDataHelp" class="form-text">Enter the data to be analyzed for Parkinson's disease prediction.</div>
            </div>
            <button type="submit" class="btn btn-primary"><i class="fas fa-paper-plane"></i> Submit</button>
          </form>

          {% if message %}
            {% if message_type == 'danger' %}
              <div class="alert-message alert-danger">
                <p>{{ message }}</p>
              </div>
            {% elif message_type == 'success' %}
              <div class="alert-message alert-success">
                <p>{{ message }}</p>
              </div>
            {% else %}
              <div class="alert-message alert-proper">
                <p>{{ message }}</p>
              </div>
            {% endif %}
          {% endif %}
        </div>
      </div>
      <div class="footer">
        <p>&copy; 2024 Parkinson Disease Detection. All rights reserved. | <a href="#">Privacy Policy</a> | <a href="#">Terms of Service</a></p>
      </div>
      <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/js/bootstrap.bundle.min.js" integrity="sha384-ENjdO4Dr2bkBIFxQpeoTz1HIcje39Wm4jDKdf19U8gI4ddQ3GYNS7NTKfAdVQSZe" crossorigin="anonymous"></script>
    </body>
    </html>
    '''
    return render_template_string(html, message=output, message_type=message_type)

if __name__ == "__main__":
    app.run(debug=True)
