#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.
By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import pandas as pd
import os
import sys
import argparse
import glob
import time

import caffe
from number_for_lcd import NumberDrawer

def main(argv):
    pycaffe_dir = os.path.dirname(__file__)

    image_dims = [28, 28]
    mean, channel_swap = None, None
    force_grayscale = True
    caffe.set_mode_cpu()
    print("CPU mode")
    model_def = os.path.join(pycaffe_dir, "../examples/mnist/lenet_50.prototxt")
    pretrained_model = os.path.join(pycaffe_dir,"../examples/mnist/lenet_iter_10000_50.caffemodel")
    input_scale = None
    raw_scale = 255.0
    print("make classifier...")
    start = time.time()
    # Make classifier.
    classifier = caffe.Classifier(model_def, pretrained_model,
            image_dims=image_dims, mean=mean,
            input_scale=input_scale, raw_scale=raw_scale,
            channel_swap=channel_swap)
    print("Done in %.2f s." % (time.time() - start))
    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    input_file = os.path.join(pycaffe_dir, "../images_test/four_28x28.png")
    input_file = os.path.expanduser(input_file)
    force_grayscale = True
    print("Loading image file: %s" % input_file)
    start = time.time()
    inputs = [caffe.io.load_image(input_file, not force_grayscale)]
    print("Done in %.2f s." % (time.time() - start))
    
    print("Classifying %d inputs." % len(inputs))
    # Classify.
    center_only = True
    start = time.time()
    predictions = classifier.predict(inputs, not center_only)
    print("Done in %.2f s." % (time.time() - start))
    labels_file = os.path.join(pycaffe_dir,"../data/mnist/mnist_words.txt")
    with open(labels_file) as f:
        labels_df = pd.DataFrame([
            {
                'synset_id': l.strip().split(' ')[0],
                'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]
            }
            for l in f.readlines()
        ])
    labels = labels_df.sort('synset_id')['name'].values

    indices = (-predictions[0]).argsort()[:5]
    predicted_labels = labels[indices]

    meta = [
                (p, '%.5f' % predictions[0][i])
                for i, p in zip(indices, predicted_labels)
            ]

    print(int(meta[0][0]))
    drawer = NumberDrawer()
    drawer.draw_number(int(meta[0][0]))  
    time.sleep(1)

if __name__ == '__main__':
    main(sys.argv)#!/usr/bin/env python