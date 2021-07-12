from flask import Flask, jsonify, request, render_template
import numpy as np
import base64
import tensorflow as tf

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
    q = np.frombuffer(r, dtype=np.uint8)
    print(q.shape)
    q = np.reshape(q, (1,512, 512,1))
    q = q.astype('float32')
    interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], q)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    return jsonify(data="{}".format(output_data[0][0]))

if __name__ == '__main__':
    app.run(debug=True, threaded='True')
