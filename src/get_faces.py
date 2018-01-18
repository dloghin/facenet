# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Copyright (c) 2018 Dumi Loghin
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
import cv2
import align.detect_face
from scipy import misc

def get_square_box(x0, y0, x1, y1, max_w, max_h):
    wb = x1 - x0
    hb = y1 - y0
    if wb < hb:
        if x1 + (hb - wb) >= max_w:
            if x0 - (hb - wb) < 0:
                x0 = 0
            else:
                x0 = x0 - (hb - wb)
        else:
            x1 = x1 + (hb - wb)
    elif wb > hb:
        if y1 + (wb - hb) >= max_h:
           if y0 - (wb - hb) < 0:
              y0 = 0
           else:
              y0 = y0 - (wb - hb)
        else:
           y1 = y1 + (vb - hb)

    return [x0, y0, x1, y1]

def main(args):
    if len(args) < 2:
        print("Usage: " + args[0] + " <image>")
        return
    
    img_path = args[1]

    with tf.Graph().as_default():      
        with tf.Session() as sess:
            
            np.random.seed(1234)

            # 0. Read image
            try:
                img = misc.imread(img_path)
            except (IOError, ValueError, IndexError) as e:
                errorMessage = '{}: {}'.format(img_path, e)
                print(errorMessage)
                return

            if img.ndim < 2:
                print('Unable to align "%s"' % image_path)
                return
            elif img.ndim == 2:
                img = facenet.to_rgb(img)
            img = img[:,:,0:3]
            [h, w] = np.asarray(img.shape)[0:2]

            # 1. Detect Face
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            
            minsize = 20 # minimum size of face
            threshold = [ 0.8, 0.85, 0.85 ]  # three steps's threshold
            factor = 0.709 # scale factor
            
            bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            faces = []
            if nrof_faces > 0:                
                for i in range(nrof_faces):
                    det = np.squeeze(bounding_boxes[i,0:4])

                    #y0 = int(det[1] * h)
                    #y1 = int(det[3] * h)
                    #x0 = int(det[0] * w)
                    #x1 = int(det[2] * w)
                    
                    [x0, y0, x1, y1] = get_square_box(int(det[0]), int(det[1]), int(det[2]), int(det[3]), w, h)
                    print(str(x0) + " " + str(y0) + " " + str(x1) + " " + str(y1))                    
                    cropped = img[y0:y1,x0:x1,:]
                    misc.imsave("roi" + str(i) + ".png", cropped)
                    scaled = misc.imresize(cropped, (160, 160), interp='bilinear')
                    faces.append(scaled)
            
            # 2. Recognize Face
  
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model('../models/facenet/20170512-110547/20170512-110547.pb')
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            feed_dict = { images_placeholder:faces, phase_train_placeholder:False }
            emb_array = sess.run(embeddings, feed_dict=feed_dict)
                       
            with open('../models/lfw_classifier.pkl', 'rb') as infile:
                (model, class_names) = pickle.load(infile)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                                    
if __name__ == '__main__':
    main(sys.argv)
