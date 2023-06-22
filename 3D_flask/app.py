import os
import shutil
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from efficientnet.tfkeras import EfficientNetB4

model=load_model('efficient_model (1).h5') 

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = np.expand_dims(img, axis=0)
    return img

app = Flask(__name__, template_folder="templateFiles", static_folder= "staticFiles")
app.config['UPLOAD_FOLDER'] = 'uploads' 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

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
   

    image= preprocess(file_path)
    prediction= model.predict(image)
    predicted_class = np.argmax(prediction[0])
    
    folders = ['OK', 'blobs', 'cracks', 'stringing', 'spaghetti', 'under exstrosion']
    result= folders[predicted_class]

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
