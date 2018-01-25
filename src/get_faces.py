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

sys.path.append("/home/ubuntu/git/tensorflow-face-detection")
from utils import label_map_util
from utils import visualization_utils_color as vis_util

''' Get a square bounding box by extending the existing box 
into a single direction. '''
def get_square_box_v1(x0, y0, x1, y1, max_w, max_h):
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

''' Get a square bounding box by extending the existing box 
both to the left and right, or both to the top and bottom, 
as needed. '''
def get_square_box(x0, y0, x1, y1, max_w, max_h):
    wb = x1 - x0
    hb = y1 - y0
    if wb < hb:
        d0 = (hb - wb) // 2
        if 2 * d0 < hb - wb:
            d1 = d0 + 1
        else:
            d1 = d0
        if x1 + d1 >= max_w:
            d1 = max_w - x1 - 1
            d0 = hb - wb - d1
            if x0 - d0 < 0:
                x0 = 0
                x1 = max_w - 1
                return get_square_box(x0, y0, x1, y1, max_w, max_h)
            else:
                x0 = x0 - d0
                x1 = x1 + d1
        else:
            x1 = x1 + d1
            if x0 - d0 < 0:
                x0 = 0
                return get_square_box(x0, y0, x1, y1, max_w, max_h)
            else:
                x0 = x0 - d0

    elif wb > hb:
        d0 = (wb - hb) // 2
        if 2 * d0 < wb - hb:
            d1 = d0 + 1
        if y1 + d1 >= max_h:
            d1 = max_h - y1 - 1
            d0 = wb - hb - d1
            if y0 - d0 < 0:
                y0 = 0
                y1 = max_h - 1
                return get_square_box(x0, y0, x1, y1, max_w, max_h)
            else:
                y0 = y0 - d0
                y1 = y1 + d1
        else:
            y1 = y1 + d1
            if y0 - d0 < 0:
                y0 = 0
                return get_square_box(x0, y0, x1, y1, max_w, max_h)
            else:
                y0 = y0 - d0

    return [x0, y0, x1, y1]


def main(args):
    if len(args) < 2:
        print("Usage: " + args[0] + " <image>")
        return
    
    img_path = args[1]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default():      
        with tf.Session(config = config) as sess:
            
            np.random.seed(666)

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
            elif len(img.shape) > 2 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            [h, w] = np.asarray(img.shape)[0:2]

            # 1. Detect Face
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            
            minsize = 20 # minimum size of face
            threshold = [ 0.8, 0.85, 0.85 ]  # three steps's threshold
            factor = 0.709 # scale factor
            
            bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            faces = []
            boxes = []
            if nrof_faces > 0:                
                for i in range(nrof_faces):
                    det = np.squeeze(bounding_boxes[i,0:4])

                    #y0 = int(det[1] * h)
                    #y1 = int(det[3] * h)
                    #x0 = int(det[0] * w)
                    #x1 = int(det[2] * w)
                    x0 = max(int(det[0]) - 20, 0)
                    x1 = min(int(det[2]) + 20, w-1)
                    y0 = max(int(det[1]) - 20, 0)
                    y1 = min(int(det[3]) + 20, h-1)
                    
                    [x0, y0, x1, y1] = get_square_box(x0, y0, x1, y1, w, h)
                    print(str(x0) + " " + str(y0) + " " + str(x1) + " " + str(y1))                    
                    cropped = img[y0:y1,x0:x1,:]
                    scaled = misc.imresize(cropped, (160, 160), interp='bilinear')
                    prew = facenet.prewhiten(scaled)
                    faces.append(prew)
                    boxes.append([x0, y0, x1, y1])
                    misc.imsave("roi" + str(i) + ".png", prew)
            
            # 2. Recognize Face
  
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model('../models/facenet/20170512-110547/20170512-110547.pb')
            # facenet.load_model('../models/facenet/20170511-185253/20170511-185253.pb')
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            emb_array = np.zeros((len(faces), embedding_size))
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            feed_dict = { images_placeholder:faces, phase_train_placeholder:False }
            emb_array[:,:] = sess.run(embeddings, feed_dict=feed_dict)

            # Load embeddings from file and concatenate with computed embeddings
            # with open('../models/emb_array.bin', 'rb') as infile:
            #    emb_array_cls = pickle.load(infile)
            # print(emb_array_cls)
            # emb_arrys = np.concatenate((emb_array, emb_array_cls), axis=0)

            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            with open('../models/facenet/lfw_classifier-20170512-110547.pkl', 'rb') as infile:
#            with open('../models/facenet/lfw_classifier-20170511-185253.pkl', 'rb') as infile:
                (model, class_names) = pickle.load(infile)

                predictions = model.predict_proba(emb_array)

# Print all prediction in sorted order
#                sorted_class_indices = np.argsort(predictions, axis=1)
#                for i in range(len(predictions)):
#                    for j in range(len(class_names)):
#                        print('%.4f %s' % (predictions[i][sorted_class_indices[i][j]], class_names[sorted_class_indices[i][j]]))
#                    print("----------")

#                print(predictions)

                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                #predictions = model.predict(emb_array)
                #best_class_indices = predictions
                #best_class_probabilities = predictions
                
                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    vis_util.draw_bounding_box_on_image_array(img, boxes[i][1], boxes[i][0], boxes[i][3], boxes[i][2], "red", 3, [class_names[best_class_indices[i]], "{:.3f}".format(best_class_probabilities[i])], False)

            cv2.imwrite("img.png", img)

if __name__ == '__main__':
    main(sys.argv)
