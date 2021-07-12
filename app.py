from flask import Flask, jsonify, request, render_template
import numpy as np
import base64
import tflite

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
    print(q.shape)
    q = np.reshape(q, (512, 512))
    print(q.shape)
    interpreter = tflite.Interpreter("converted_model.tflite")
    print("Interpreter Initialized !!")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], q)
    interpreter.invoke()
    print("Interpreter Invoked")
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    return "ad"

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
