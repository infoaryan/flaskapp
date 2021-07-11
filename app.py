from flask import Flask, jsonify, request, render_template
import numpy as np


# app
app = Flask(__name__)

# routes
@app.route('/')
def predict():
    return render_template("index.html")

@app.route('/prediction',methods=["POST"])
def prediction():
    img = request.files['img']
    return model.summary()

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
