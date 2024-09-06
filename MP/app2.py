from flask import Flask, request, redirect, url_for
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'photo1' not in request.files or 'photo2' not in request.files:
            return 'No file part'
        photo1 = request.files['photo1']
        photo2 = request.files['photo2']
        if photo1.filename == '' or photo2.filename == '':
            return 'No selected file'
        if photo1 and photo2:
            photo1.save(os.path.join(app.config['UPLOAD_FOLDER'], photo1.filename))
            photo2.save(os.path.join(app.config['UPLOAD_FOLDER'], photo2.filename))
            return 'Files successfully uploaded'
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PARKINSON DISEASE DETECTION SYSTEM</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: #1a1a1a;
            background: linear-gradient(135deg, #6db3f2 0%, #1e69de 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            background: rgba(255, 255, 255, 0.95);
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
            width: 90%;
            max-width: 500px;
            text-align: center;
        }
        h1 {
            font-size: 2em;
            margin-bottom: 25px;
            color: #1e69de;
        }
        label {
            display: block;
            margin-top: 25px;
            margin-bottom: 15px;
            font-weight: bold;
            font-size: 1.2em;
            color: #333;
        }
        input[type="file"] {
            display: block;
            margin: 0 auto 25px;
            padding: 12px;
            border: 1px solid #ccc;
            border-radius: 5px;
            width: 100%;
            max-width: 350px;
            font-size: 1em;
            background-color: #f9f9f9;
            transition: all 0.3s;
        }
        input[type="file"]:hover {
            border-color: #1e69de;
        }
        button {
            background-color: #1e69de;
            color: white;
            border: none;
            padding: 15px 35px;
            border-radius: 50px;
            cursor: pointer;
            font-size: 1.1em;
            transition: background-color 0.3s;
        }
        button:hover {
            background-color: #1557b8;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>PARKINSON DISEASE DETECTION</h1>
        <form action="http://localhost:8080/" method="post" enctype="multipart/form-data">
            <label for="photo1">Upload Spiral Drawing:</label>
            <input type="file" id="photo1" name="photo1" accept="image/*" required>
            <label for="photo2">Upload Wave Drawing:</label>
            <input type="file" id="photo2" name="photo2" accept="image/*" required>
            <button type="submit">Upload</button>
        </form>
    </div>
</body>
</html>
'''

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(port=8080, debug=True)
