#import pandas as pd
from flask import Flask, jsonify, request, render_template
#import pickle
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# load model
#model = pickle.load(open('model.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/')
def predict():
    return render_template("index.html")

@app.route('/prediction',methods=["POST"])
def prediction():
    img = request.files['img']
    img.save("img.jpg")
    image = cv2.imread("img.jpg")
    image = cv2.resize(image, (512,512))
    image = np.reshape(image, (1,512,512,3))
    image = image.astype('float32')
    image/=255.
    result = model.predict(image)
    pas = np.argmax(np.argmax(result))
    return render_template("prediction.html",data = pas)

if __name__ == '__main__':
    model = load_model('Resnet_temp.hdf5')
    app.run(port = 5000, debug=True)