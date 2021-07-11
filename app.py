from flask import Flask, jsonify, request, render_template
import numpy as np


# app
app = Flask(__name__)

# routes
@app.route('/')
def predict():
    return render_template("index.html")

@app.route('/prediction',methods=["GET","POST"])
def prediction():
    print("Got into method")
    print(request)
    data = request['output_size']
    print(data)
    data1 = request['input_image']
    print(data1)
    echo "ad"
    return "gotit"

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
