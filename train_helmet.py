import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

import helmet
from helmet import HelmetConfig
from helmet import train
from helmet import detect_and_color_splash

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

COMMAND = 'train'   # Update to 'train' or 'inference'
HELMET_DIR = os.path.join(ROOT_DIR, "/home/eleensmathew/Helmet_Mask_RCNN/person on bike.v7i.coco")  # Update - Required for'train'
#IMAGE_FILE = os.path.join(ROOT_DIR, "datasets/helmet/va/VID_20200315_181352_frame0.jpg") # Update - Required for 'inference'
VIDEO_FILE = None # either IMAGE_FILE or VIDEO_FILE required

# Update to 'coco' or 'last' - use the last generated weights or specify the weights file path
# WEIGHTS = 'last'  
# WEIGHTS = os.path.join(MODEL_DIR, "helmet20200509T1224/mask_rcnn_helmet_0070.h5")  # Specific model weights to be used

WEIGHTS = 'coco'    # To continue the training from last checkpoint 
#WEIGHTS = os.path.join(ROOT_DIR, "mask_rcnn_helmet.h5") 

# Set to False if needs to be 'Motorcyclist_with_Helmet' or 'without_Helmet' instead
helmet.DEFAULT_LABEL_HELMET_SEPARATE = True 

helmet.NBR_OF_EPOCHS = 115

# Configurations

class InferenceConfig(HelmetConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

if COMMAND == "train":
    config = HelmetConfig()
    helmet.config = config
    config.display()
# else:
#     config = InferenceConfig()


# Create model
if COMMAND == "train":
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)
    print ('Model loaded for training')

if COMMAND == 'train' :
    # Select weights file to load
    if WEIGHTS == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif WEIGHTS == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif WEIGHTS == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = WEIGHTS
    print (f'Weights file selected for {WEIGHTS}')

    # Load weights
    print("Loading weights ", weights_path)
    if WEIGHTS == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    print(f'{weights_path} weights loaded')

if COMMAND == "train":
    train(model, HELMET_DIR)
