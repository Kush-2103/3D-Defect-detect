import os
import shutil
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model


# model=load_model()  put the h5 model here

app = Flask(__name__, template_folder="templateFiles", static_folder= "staticFiles")
app.config['UPLOAD_FOLDER'] = 'uploads' 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)

    try:
        shutil.rmtree(app.config['UPLOAD_FOLDER'])
    except:
        pass

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    uploaded_file.save(file_path)
    # Perform defect detection using your model

    # result = detect_defect(file_path)

    result = "Defect detected or not"
    return render_template('index.html', result=result)
def detect_defect(file_path):

    # Load and preprocess the image
    

    # Perform prediction
    #prediction = model.predict(img)
    # Replace this with your logic to interpret the prediction and determine if a defect is detected
    # if prediction[0][0] > 0.5:
    #     result = "Defect detected"
    # else:
    #     result = "No defect detected"

    return "Detect defect or not will be displayed here"

if __name__ == '__main__':
    app.run(debug=True)
