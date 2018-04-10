import os, sys, time, cv2, plots
import numpy as np
from plots import bokeh_colors, imshow


def annotateImage(im, S, source='YOLO'):
    h, w, ch = im.shape
    n = len(S)
    c = bokeh_colors(n)
    c = [255, 255, 255]
    thick = round(h * .003)
    if source is 'YOLO':
        for i in np.arange(n):
            a = S[i]
            left, top = a['topleft']['x'], a['topleft']['y']
            right, bot = a['bottomright']['x'], a['bottomright']['y']
            cv2.rectangle(im, (left, top), (right, bot), c, thick * 2)
            cv2.putText(im, '%s %.0f%%' % (a['label'], a['confidence'] * 100), (left, top - 12), 0, 1e-3 * h, c, thick)
    return im


def mainYOLO():
    from darkflow.net.build import TFNet
    path = '/Users/glennjocher/'
    fname = path + 'Downloads/IMG_4122.JPG'
    options = {'model': path + 'darkflow/cfg/tiny-yolo-voc.cfg', 'load': path + 'darknet/yolov2-tiny-voc.weights',
               'threshold': 0.3}
    options = {'model': path + 'darkflow/cfg/yolo.cfg', 'load': path + 'darkflow/yolo.weights', 'threshold': 0.3}
    tfnet = TFNet(options)

    # IMAGE
    im = cv2.imread(fname)  # native BGR
    tic = time.time()
    result = tfnet.return_predict(im)  # wants BGR
    print('Single image complete. %.3f s elapsed.' % (time.time() - tic))
    im = annotateImage(im, result, source='YOLO')
    imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

    # VIDEO
    fname = path + 'Downloads/DATA/VSM/2018.3.30/IMG_4238.m4v'
    cap = cv2.VideoCapture(fname)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    out = cv2.VideoWriter(fname + '.yolo.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
    for i in np.arange(90):
        success, im = cap.read()  # native BGR
        if success:
            tic = time.time()
            result = tfnet.return_predict(im)  # wants BGR
            print('Frame %g/%g ... %.3fs.' % (i, frame_count, time.time() - tic))
            if any(result):
                im = annotateImage(im, result, source='YOLO')
            out.write(im)  # wants BGR
            # imshow(im)
        else:
            break
    cap.release()
    out.release()

    return None
    # ./darknet detect cfg/yolov3.cfg yolov3.weights /Users/glennjocher/Downloads/IMG_4122.JPG
    # ./darknet detect cfg/yolov2-tiny.cfg yolov2-tiny.weights /Users/glennjocher/Downloads/IMG_4122.JPG


def mainTF():
    import tensorflow as tf
    tic = time.time()

    # This is needed since the notebook is stored in the object_detection folder.
    TF_PATH = '/Users/glennjocher/tensorflow/models/research/object_detection/'
    sys.path.append('/Users/glennjocher/tensorflow/models/research/object_detection/')
    sys.path.append('/Users/glennjocher/tensorflow/models/research/')
    sys.path.append('..')
    from object_detection.utils import ops as utils_ops
    from utils import label_map_util
    from utils import visualization_utils as vis_util

    # What model to download.
    # MODEL_NAME = 'mask_rcnn_inception_v2_coco_2018_01_28'
    # MODEL_NAME = 'faster_rcnn_inception_v2_coco_2018_01_28'
    # MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'

    # List of the strings that is used to add correct label for each box.
    NUM_CLASSES = 90

    # Load a (frozen) Tensorflow model into memory.
    tfgraph = tf.Graph()
    with tfgraph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(MODEL_NAME + '/frozen_inference_graph.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this
    # corresponds to `airplane`.  Here we use internal utility functions, but
    # anything that returns a dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(TF_PATH + 'data/mscoco_label_map.pbtxt')
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Detection
    # image1.jpg
    # image2.jpg
    TEST_IMAGE_PATHS = [os.path.join(TF_PATH + 'test_images/', 'image{}.jpg'.format(i)) for i in range(1, 3)]
    TEST_IMAGE_PATHS = ['/Users/glennjocher/Downloads/taya.jpg', '/Users/glennjocher/Downloads/IMG_4122.JPG']
    print('Loaded TF. %.1f s elapsed.' % (time.time() - tic))

    def run_inference_for_single_image(im, graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                T = {}  # tensor dictionary
                for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes',
                            'detection_masks']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        T[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                if 'detection_masks' in T:
                    # The following processing is only for single im
                    boxes = tf.squeeze(T['detection_boxes'], 0)
                    masks = tf.squeeze(T['detection_masks'], 0)
                    # Reframe is required to translate mask from box coordinates to im coordinates and fit the im size.
                    j = tf.cast(T['num_detections'][0], tf.int32)
                    boxes = tf.slice(boxes, [0, 0], [j, -1])
                    masks = tf.slice(masks, [0, 0, 0], [j, -1, -1])
                    masks_reframed = utils_ops.reframe_box_masks_to_image_masks(masks, boxes, im.shape[0], im.shape[1])
                    masks_reframed = tf.cast(tf.greater(masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    T['detection_masks'] = tf.expand_dims(masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                D = sess.run(T, feed_dict={image_tensor: np.expand_dims(im, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                D['num_detections'] = int(D['num_detections'][0])
                D['detection_classes'] = D['detection_classes'][0].astype(np.uint8)
                D['detection_boxes'] = D['detection_boxes'][0]
                D['detection_scores'] = D['detection_scores'][0]
                if 'detection_masks' in D:
                    D['detection_masks'] = D['detection_masks'][0]
        return D

    for image_path in TEST_IMAGE_PATHS:
        im = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)  # TF wants RGB

        # Actual detection.
        tic = time.time()
        D = run_inference_for_single_image(im, tfgraph)
        print('Single image complete. %.1f s elapsed.' % (time.time() - tic))

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(im,
                                                           D['detection_boxes'],
                                                           D['detection_classes'],
                                                           D['detection_scores'],
                                                           category_index,
                                                           instance_masks=D.get('detection_masks'),
                                                           use_normalized_coordinates=True,
                                                           line_thickness=6)
        plots.imshow(im)


mainYOLO()
