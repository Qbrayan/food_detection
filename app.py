import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw
import skimage

from flask import Flask, request, jsonify, make_response, render_template, url_for
from settings import *
import pathlib
import requests

#PACKAGE_ROOT = pathlib.Path(__file__).resolve()
#ROOT_DIR = os.path.abspath("")
#print(PACKAGE_ROOT)
print(MODEL_DIR)

# Import Mask RCNN
#sys.path.append(PACKAGE_ROOT)  # To find local version of the library

app = Flask(__name__)

#####################################################################################
################################### ROUTES ##########################################
#####################################################################################

from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
from mrcnn.model import MaskRCNN
import tensorflow as tf
import keras.backend as K

@app.route('/', methods=['GET'])
def healthcheck():
    return ("OK", 200)

class FoodModelConfig(Config):
    """Configuration for training on the cigarette butts dataset.
    Derives from the base Config class and overrides values specific
    to the cigarette butts dataset.
    """
    # Give the configuration a recognizable name
    NAME = "foodmodel"

    # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # background + 1 (classes of food)

    # All of our training images are 512x512
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # You can experiment with this number to see if it improves training
    STEPS_PER_EPOCH = 500

    # This is how often validation is run. If you are using too much hard drive space
    # on saved models (in the MODEL_DIR), try making this value larger.
    VALIDATION_STEPS = 5
    
    # Matterport originally used resnet101, but I downsized to fit it on my graphics card
    BACKBONE = 'resnet50'

    # To be honest, I haven't taken the time to figure out what these do
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    MAX_GT_INSTANCES = 50 
    POST_NMS_ROIS_INFERENCE = 500 
    POST_NMS_ROIS_TRAINING = 1000 
    
config = FoodModelConfig()
#config.display()

class InferenceConfig(FoodModelConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    DETECTION_MIN_CONFIDENCE = 0.85
    
inference_config = InferenceConfig()

tf_config = tf.ConfigProto(
    device_count={'CPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)


sess = tf.Session(config=tf_config)
graph = tf.get_default_graph()
K.set_session(sess)

os.makedirs(os.path.join(MODELS_ROOT,'trained-models'),exist_ok=True)

model_path ="https://github.com/Qbrayan/food_detection/releases/download/0.1.0/mask_rcnn_foodmodel_0030.h5"

q = requests.get(model_path)

with open(MODEL_DIR, 'wb') as f:
    f.write(q.content)


# Recreate the model in inference mode
model = MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)



model.load_weights(MODEL_DIR, by_name=True) 





class_names = ['BG', 'Chicken', 'Eba', 'Fish', 'Rice', 'Bread' ]


#for image_path in image_paths:
# img = skimage.io.imread(IMG_DIR)
# img_arr = np.array(img)
# results = model.detect([img_arr], verbose=1)
# r = results[0]
#visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], 
 #                           class_names, r['scores'], figsize=(5,5))

#ans = visualize.apply_mask(img, r['masks'], color=None, alpha=0.5)
#print(r['class_ids'])



@app.route('/predict',methods=['POST', 'GET'])
def predict():
    pass

@app.route('/predict_api', methods=['POST'])
def predict_api():
    image = Image.open(request.files['image'])
    load_image = image.convert('RGB')
    load_image.save('im.jpg', format="JPEG")
    img = skimage.io.imread('im.jpg')
    img_arr = np.array(img)
    global sess
    global graph
    with graph.as_default():
        K.set_session(sess)
        results = model.detect([img_arr], verbose=1)
    r = results[0]
    cats=[]
    for i in r['class_ids']:
        cats.append(class_names[i])
    return jsonify({'classes':cats})



#####################################################################################
############################### INTIALIZATION CODE ##################################
#####################################################################################

if __name__ == '__main__':
    try:
        port = int(PORT)
    except Exception as e:
        print("Failed to bind to port {}".format(PORT))
        port = 80

    app.run(port=port , debug = True)

    # disable logging so it doesn't interfere with testing
    app.logger.disabled = True
