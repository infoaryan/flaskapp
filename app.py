from flask import Flask, jsonify, request, render_template
import numpy as np
import base64

# app
app = Flask(__name__)

# routes
@app.route('/')
def predict():
    return render_template("index.html")

@app.route('/prediction',methods=["GET","POST"])
def prediction():
    data = request.json['input_image']
    r = base64.b64decode(data)
    q = np.frombuffer(r, dtype=np.float64)
    print(q)
    return "ad"

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
