from flask import Flask, jsonify, request, render_template
import numpy as np
import base64
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.cm as cm
import json


# app
app = Flask(__name__)

# routes
@app.route('/')
def predict():
    return render_template("index.html")

@app.route('/prediction',methods=["GET","POST"])
def prediction():
    print(request)
    data = request.json['input_image']
    r = base64.b64decode(data)
    q = np.frombuffer(r, dtype=np.uint8)
    print(q.shape)
    q = np.reshape(q, (1,512, 512,1))
    print(q.shape)
    q = q.astype('float32')
    net = tf.keras.models.load_model('model.hdf5')
    print("Model Loaded !!")
    conv_layer = net.get_layer("block7a_project_conv")
    heatmap_model = tf.keras.models.Model([net.inputs], [conv_layer.output, net.output])
    # Get gradient of the winner class w.r.t. the output of the (last) conv. layer
    print("Heatmodel generated !!!")
    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(q)
        loss = predictions[:, np.argmax(predictions[0])]
        grads = gtape.gradient(loss, conv_output)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
    print(predictions)
    predicted_class = np.argmax(predictions[0])
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    print(heatmap.shape)
    #Now the generated heatmap has to move through colourizing procedure
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap[0])
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    #interpreter.set_tensor(input_details[0]['index'], q)
    #interpreter.invoke()
    print("Heatmap Generated-------------------------------")
    #output_data = interpreter.get_tensor(output_details[0]['index'])
    print(jet_heatmap.shape)

    bdata = base64.b64encode(jet_heatmap).decode()
    #return jsonify(data="{}".format(bdata))
    payload = {"predicted_class" : "{}".format(predicted_class),"saliency_map" : bdata}
    return json.dumps(payload)

if __name__ == '__main__':
    app.run(threaded = 'True', debug=True)
