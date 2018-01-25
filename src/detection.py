import time
import numpy as np
import tensorflow as tf
import sys
from matplotlib import pyplot as plt
from PIL import Image
sys.path.append("/home/ubuntu/git/models/research")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# Paths
model_path = '../models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'

labels_path = '../../models/research/object_detection/data/mscoco_label_map.pbtxt'

num_classes = 90

# Load a tensorflow model into memory
def init_model(model_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph


# Load label map
def init_label_map(labels_path, num_classes):
    label_map = label_map_util.load_labelmap(labels_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    return category_index


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# Detect objects in the image
def detect(image_np, detection_graph):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            return boxes, scores, classes, num


def visualize(boxes, scores, classes, image_np, category_index):
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    plt.imshow(image_np)
    plt.show()

def main(args):
    if len(args) < 2:
        print("Usage: " + args[0] + " <image>")
        return
	
    img_path = args[1]

    # Load the detection model
    startTime = time.time()
    detection_graph = init_model(model_path)
    print('Loading the model took {} s'.format(time.time() - startTime))

    # Read the input image
    image = Image.open(img_path)
    image_np = load_image_into_numpy_array(image)

    # Run the detector
    startTime = time.time()
    boxes, scores, classes, _ = detect(image_np, detection_graph)
    print('Detection took {} s'.format(time.time() - startTime))
    print(classes)
    print(scores)

    # Visualize results
    category_index = init_label_map(labels_path, num_classes)
    visualize(boxes, scores, classes, image_np, category_index)

if __name__ == '__main__':
    main(sys.argv)


