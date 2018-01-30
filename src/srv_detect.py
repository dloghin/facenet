import socket
import re
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
import threading

HOST = ''
PORT = 8888

data_ready = False
data_images = []

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


def client_thread(conn):
    # Client should first send a list of images of the form:
    # <img_type0,size0>;<img_type1,size1>;...
    data = conn.recv(2048).split(";")
    n = 0
    for token in data:
        image = token.split(",")
        conn.send("OK")
        img_data = conn.recv(int(image[1]))
        img_name = "img" + str(n) + "." + image[0]
        with open(img_name, 'wb') as f:
            f.write(img_data)
            f.close()
        n = n + 1
        data_images.append(img_name)
    conn.close()
    data_ready = True
        
def main(args):
    # networking
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((HOST, PORT))
    except socket.error as msg:
        print("Socket bind failed. Error code : " + str(msg[0]) + ", message " + msg[1])
        return
    sock.listen(10)
    
    # ai
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Graph().as_default():      
        with tf.Session(config = config) as sess:
            
            np.random.seed(666)

            # 1. Detect Face
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
            minsize = 20 # minimum size of face
            threshold = [ 0.8, 0.85, 0.85 ]  # three steps's threshold
            factor = 0.709 # scale factor

            # 2. Recognize Face
  
            # Load facenet
            print('Loading feature extraction model')
            facenet.load_model('../models/facenet/20170512-110547/20170512-110547.pb')
            # facenet.load_model('../models/facenet/20170511-185253/20170511-185253.pb')
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]            

            # Load classifier
            print('Loading face classifier')
            with open('../models/facenet/lfw_classifier-20170512-110547.pkl', 'rb') as infile:
#           with open('../models/facenet/lfw_classifier-20170511-185253.pkl', 'rb') as infile:
                (model, class_names) = pickle.load(infile)

            while True:
                print("Waiting for a connection...")
                conn, addr = sock.accept()
                client_thread(conn)
                total_boxes = bulk_detect_face(data_images, 0.25, pnet, rnet, onet, threshold, factor)
                n = 0
                faces = []
                for image in data_images:
                    bounding_boxes = total_boxes[n]
                    n = n + 1
                    det = np.squeeze(bounding_boxes[i,0:4])
                    x0 = max(int(det[0]) - 20, 0)
                    x1 = min(int(det[2]) + 20, w-1)
                    y0 = max(int(det[1]) - 20, 0)
                    y1 = min(int(det[3]) + 20, h-1)
                    [x0, y0, x1, y1] = get_square_box(x0, y0, x1, y1, w, h)
                    cropped = img[y0:y1,x0:x1,:]
                    scaled = misc.imresize(cropped, (160, 160), interp='bilinear')
                    prew = facenet.prewhiten(scaled)
                    faces.append(prew)
                    boxes.append([x0, y0, x1, y1])
                    # misc.imsave("roi" + str(i) + ".png", prew)

                emb_array = np.zeros((len(faces), embedding_size))
                feed_dict = { images_placeholder:faces, phase_train_placeholder:False }
                emb_array[:,:] = sess.run(embeddings, feed_dict=feed_dict)
                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                

if __name__ == "__main__":
	main(sys.argv)
