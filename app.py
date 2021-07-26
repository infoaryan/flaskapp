from flask import Flask, request, render_template
from numpy import frombuffer,reshape, argmax, maximum,max, uint8, arange
from base64 import b64decode, b64encode
from tensorflow.keras.models import load_model
from tensorflow.keras.backend import mean
from tensorflow import GradientTape, reduce_mean, multiply
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
    r = b64decode(data)
    q = frombuffer(r, dtype=uint8)
    print(q.shape)
    q = reshape(q, (1,512, 512,1))
    print(q.shape)
    q = q.astype('float32')
    heatmap_model = load_model('model.hdf5',compile = False)
    #print("Model Loaded !!")
    #conv_layer = net.get_layer("block7a_project_conv")
    #heatmap_model = tf.keras.models.Model([net.inputs], [conv_layer.output, net.output])
    # Get gradient of the winner class w.r.t. the output of the (last) conv. layer
    print("Heatmodel generated !!!")
    with GradientTape() as gtape:
        conv_output, predictions = heatmap_model(q)
        loss = predictions[:, argmax(predictions[0])]
        grads = gtape.gradient(loss, conv_output)
        pooled_grads = mean(grads, axis=(0, 1, 2))
    predicted_class = argmax(predictions[0])
    del q
    del heatmap_model
    if(predicted_class==1):
        payload = {"predicted_class" : "{}".format(predicted_class),"saliency_map" : ""}
        return json.dumps(payload)
    heatmap = reduce_mean(multiply(pooled_grads, conv_output), axis=-1)
    heatmap = maximum(heatmap, 0)
    max_heat = max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    print(heatmap.shape)
    #Now the generated heatmap has to move through colourizing procedure
    # Rescale heatmap to a range 0-255
    heatmap = uint8(255 * heatmap[0])
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    # Use RGB values of the colormap
    jet_colors = jet(arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    #interpreter.set_tensor(input_details[0]['index'], q)
    #interpreter.invoke()
    print("Heatmap Generated-------------------------------")
    #output_data = interpreter.get_tensor(output_details[0]['index'])
    print(jet_heatmap.shape)

    bdata = b64encode(jet_heatmap).decode()
    payload = {"predicted_class" : "{}".format(predicted_class),"saliency_map" : bdata}
    return json.dumps(payload)

if __name__ == '__main__':
    app.run(threaded = 'True', debug=True)
