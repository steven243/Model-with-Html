import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model_save.h5'
pesos_path = 'models/model_pesos.h5'

model = load_model(MODEL_PATH)
model.load_weights(pesos_path)

# Load your trained model
#model._make_predict_function()          # Necessary

print('Model loading...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
from tensorflow.keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#graph = tf.get_default_graph()

print('Model loaded. Started serving...')

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    original = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    
    # Convert the PIL image to a numpy array
    # IN PIL - image is in (width, height, channel)
    # In Numpy - image is in (height, width, channel)
    numpy_image = np.array(image.img_to_array(original))

    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.expand_dims(numpy_image, axis=0)

    print('PIL image size = ', original.size)
    print('NumPy image size = ', numpy_image.shape)
    print('Batch image  size = ', image_batch.shape)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #processed_image = preprocess_input(image_batch, mode='caffe')
    
    #with graph.as_default():    
        
    #preds = model.predict(processed_image)
    prediction = model.predict_classes(image_batch)
    
    print('Deleting File at Path: ' + img_path)

    os.remove(img_path)

    print('Deleting File at Path - Success - ')

    return prediction


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        print('Begin Model Prediction...')

        # Make prediction
        preds = model_predict(file_path, model)

        print('End Model Prediction...')

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        #result = str(pred_class[0][0][1])             # Convert to string
        
        result = str(preds)
        
        print(result)
        
        if result == "[0]":
            result = " Aplastado" 
        elif result == "[1]":
            result =  " Arrugado"
        elif result == "[2]":
            result =  " Broca"
        elif result == "[3]":
            result =  " Cardenillo"
        elif result == "[4]":
            result =  " Cristalizado"
        elif result == "[5]":
            result =  " Flojo"
        elif result == "[6]":
            result =  " Inmaduro"
        elif result == "[7]":
            result =  " Mantequillo"
        elif result == "[8]":
            result =  " Mordido o cortado"
        elif result == "[9]":
            result =  " Negro"
        elif result == "[10]":
            result =  " Reposado"
        elif result == "[11]":
            result =  " Sano"
        elif result == "[12]":
            result =  " Sobre secado"
        elif result == "[13]":
            result =  " Veteado"
        elif result == "[14]":
            result =  " Vinagre"
        
        return result
    return None

if __name__ == '__main__':    
    app.run(debug=False, threaded=False)

    # Serve the app with gevent
    # http_server = WSGIServer(('0.0.0.0', 5000), app)
    # http_server.serve_forever()