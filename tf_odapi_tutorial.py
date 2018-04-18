import os, sys, time, cv2, plots
import numpy as np


def annotateImageSpeed(im, str):
    h = im.shape[0]
    thick = round(h * .002)
    cv2.putText(im, str, (0, round(.05 * h)), 0, round(.001 * h), [255, 255, 255], thick, lineType=cv2.LINE_AA)
    return im


def annotateImageDF(im, r):
    h, w, ch = im.shape
    n = len(r)
    thick = round(h * .003)

    c = [255, 255, 255]
    for i in range(n):
        a = r[i]
        left, top = a['topleft']['x'], a['topleft']['y']
        right, bot = a['bottomright']['x'], a['bottomright']['y']
        cv2.rectangle(im, (left, top), (right, bot), c, thick + 2)
        cv2.putText(im, 'df %s %.0f%%' % (a['label'], a['confidence'] * 100), (left, top - round(h * .01)),
                    0, 1e-3 * h, c, thick, lineType=cv2.LINE_AA)
    return im


def annotateImageDN(im, r):
    h, w, ch = im.shape
    n = len(r)
    thick = round(h * .003)

    c = [0, 0, 0]  # orange
    for i in range(n):
        a = r[i]
        b = a[2]
        left, top = int(b[0] - b[2] / 2), int(b[1] - b[3] / 2)
        right, bot = int(b[0] + b[2] / 2), int(b[1] + b[3] / 2)
        cv2.rectangle(im, (left, top), (right, bot), c, thick + 2)
        cv2.putText(im, 'dn %s %.0f%%' % (a[0].decode('utf-8'), a[1] * 100), (left, top - round(h * .01)),
                    0, 1e-3 * h, c, thick, lineType=cv2.LINE_AA)
    return im


def mainYOLO():
    PATH = '/Users/glennjocher/darknet/'  # local path to cloned darknet repo
    model = 'yolov2-tiny'
    cfgPATH = PATH + 'cfg/' + model + '.cfg'
    weightsPATH = PATH + model + '.weights'
    sys.path.append(PATH)

    # Darkflow
    from darkflow.net.build import TFNet
    options = {'model': cfgPATH, 'load': weightsPATH, 'threshold': 0.6}
    tfnet = TFNet(options)
    # yolov2.224: 143ms
    # yolov2.320: 240ms
    # yolov2.416: 395ms
    # yolov2-tiny.224: 43ms
    # yolov2-tiny.416: 130ms

    # Darknet
    import darknet as dn
    net = dn.load_net(cfgPATH.encode('utf-8'), weightsPATH.encode('utf-8'), 0)
    meta = dn.load_meta(PATH.encode('utf-8') + b'cfg/coco.data')

    # # IMAGE
    # # fname = PATH + '../Downloads/IMG_4122.JPG'
    # fname = PATH + 'data/dog.jpg'
    # im = cv2.imread(fname)  # native BGR
    #
    # tic = time.time()
    # rf = tfnet.return_predict(im)  # wants BGR
    # print('%.3fs darkflow\n%s' % (time.time() - tic, rf))
    #
    # tic = time.time()
    # rn = dn.detect(net, meta, im)
    # print('%.3fs darknet\n%s' % (time.time() - tic, rn))
    #
    # im = annotateImageDF(im, rf)
    # im = annotateImageDN(im, rn)
    #
    # plots.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    # cv2.imwrite(fname + '.yolo.jpg', im)
    # # return

    # VIDEO
    fname = PATH + '../Downloads/DATA/VSM/2018.3.4/IMG_3930.MOV'
    cap = cv2.VideoCapture(fname)
    scale = 1 / 4
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(fname[:-3] + model + str(scale) + '.416.mov', cv2.VideoWriter_fourcc(*'avc1'), fps,
                          (width, height))

    dt = np.zeros((nframes, 2))
    for i in range(30 * 2):
        success, im = cap.read()  # native BGR
        if success:
            im = cv2.resize(im, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

            tic = time.time()
            rf = tfnet.return_predict(im)  # wants BGR
            dt[i, 0] = time.time() - tic
            s = '%s %5.3fs (mean %5.3fs)' % ('darkflow', dt[i, 0], dt[min(i, 1):i + 1, 0].mean())
            print('%g/%g %s' % (i, nframes, s))
            if any(rf): im = annotateImageDF(im, rf)

            tic = time.time()
            rn = dn.detect(net, meta, im)
            dt[i, 1] = time.time() - tic
            s = '%s %5.3fs (mean %5.3fs)' % ('darknet', dt[i, 1], dt[min(i, 1):i + 1, 1].mean())
            print('%g/%g %s' % (i, nframes, s))
            if any(rn): im = annotateImageDN(im, rn)

            out.write(im)  # wants BGR
            # plots.imshow(im)
        else:
            break
    cap.release()
    out.release()

    return
    # ./darknet detect cfg/yolov3.cfg yolov3.weights /Users/glennjocher/Downloads/IMG_4122.JPG
    # ./darknet detect cfg/yolov2-tiny.cfg yolov2-tiny.weights /Users/glennjocher/Downloads/IMG_4122.JPG


def mainTF(MODEL_NAME, mname):
    tic = time.time()
    import tensorflow as tf
    print('\nimport tensorflow as tf... %.1fs' % (time.time() - tic))

    # MODEL_NAME, mname = 'data/mask_rcnn_inception_v2_coco_2018_01_28', 'mask_rcnn_inception_'
    # MODEL_NAME, mname = 'data/faster_rcnn_inception_v2_coco_2018_01_28', 'faster_rcnn_inception_'
    # MODEL_NAME, mname = 'data/rfcn_resnet101_coco_2018_01_28', 'rfcn_resnet101_'
    # MODEL_NAME, mname = 'data/ssd_inception_v2_coco_2017_11_17', 'ssd_inception_'
    # MODEL_NAME, mname = 'data/ssd_mobilenet_v1_coco_2017_11_17', 'ssd_mobilenetv1_'
    # MODEL_NAME, mname = 'data/ssd_mobilenet_v2_coco_2018_03_29', 'ssd_mobilenetv2_'

    tic = time.time()
    PATH = '/Users/glennjocher/'  # local path to cloned darknet repo
    tfPATH = PATH + 'tensorflow/models/research/object_detection/'
    sys.path.append(tfPATH)
    sys.path.append(tfPATH + '..')
    from utils import ops as utils_ops
    from utils import label_map_util
    from utils import visualization_utils as vis_util

    # Loading label map
    label_map = label_map_util.load_labelmap(tfPATH + 'data/mscoco_label_map.pbtxt')
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=label_map.__sizeof__(),
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load a (frozen) Tensorflow model into memory.
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(MODEL_NAME + '/frozen_inference_graph.pb', 'rb') as fid:
        serialized_graph = fid.read()
        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name='')

    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    T = {}  # tensor dictionary
    for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            T[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    print('Load tfmodel into memory... %.1fs' % (time.time() - tic))

    # Start session
    sess = tf.Session()

    # # IMAGES
    # # fname = [os.path.join(tfPATH + 'test_images/', 'image{}.jpg'.format(i)) for i in range(1, 3)]
    # fname = ['/Users/glennjocher/Downloads/taya.jpg', '/Users/glennjocher/Downloads/IMG_4122.JPG']
    # for image_path in fname:
    #     im = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)  # TF wants RGB
    #     U = T.copy()
    #     if 'detection_masks' in U:
    #         # The following processing is only for single im
    #         boxes = tf.squeeze(U['detection_boxes'], 0)
    #         masks = tf.squeeze(U['detection_masks'], 0)
    #         # Reframe is required to translate mask from box coordinates to im coordinates and fit the im size.
    #         j = tf.cast(U['num_detections'][0], tf.int32)
    #         boxes = tf.slice(boxes, [0, 0], [j, -1])
    #         masks = tf.slice(masks, [0, 0, 0], [j, -1, -1])
    #         masks_reframed = utils_ops.reframe_box_masks_to_image_masks(masks, boxes, im.shape[0], im.shape[1])
    #         masks_reframed = tf.cast(tf.greater(masks_reframed, 0.5), tf.uint8)
    #         # Follow the convention by adding back the batch dimension
    #         U['detection_masks'] = tf.expand_dims(masks_reframed, 0)
    #
    #     # Actual detection.
    #     tic = time.time()
    #     D = sess.run(T, feed_dict={image_tensor: np.expand_dims(im, 0)})
    #     print('Inference... %.3fs' % (time.time() - tic))
    #
    #     # Visualize
    #     vis_util.visualize_boxes_and_labels_on_image_array(im, D['detection_boxes'][0],
    #                                                        D['detection_classes'][0].astype(np.uint8),
    #                                                        D['detection_scores'][0], category_index,
    #                                                        instance_masks=D.get('detection_masks'),
    #                                                        use_normalized_coordinates=True, line_thickness=8)
    #     plots.imshow(im)

    # VIDEO
    fname = PATH + 'Downloads/DATA/VSM/2018.3.4/IMG_3930.MOV'
    cap = cv2.VideoCapture(fname)
    scale = 1 / 4
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    fps = cap.get(cv2.CAP_PROP_FPS)
    nframes = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    mname += str(scale)
    out = cv2.VideoWriter(fname[:-3] + mname + '.mov', cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

    mode = 'perFrame'
    tic0 = time.time()
    nframes = 30 * 4
    dt = np.zeros(nframes)
    if mode == 'allAtOnce':
        tic = time.time()
        IM = np.empty((nframes, height, width, 3), dtype=np.uint8)
        for i in range(nframes):
            success, im = cap.read()  # native BGR
            if success:
                IM[i] = cv2.resize(im, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            else:
                break
        print('Reading %s frames all at once... %.3fs mean' % (nframes, (time.time() - tic) / nframes))

        tic = time.time()
        D = sess.run(T, feed_dict={image_tensor: np.flip(IM, axis=3)})  # TF wants RGB
        print('Inference... %.3fs mean' % ((time.time() - tic) / nframes))
        for j in range(i):
            vis_util.visualize_boxes_and_labels_on_image_array(IM[j], D['detection_boxes'][j],
                                                               D['detection_classes'][j].astype(np.uint8),
                                                               D['detection_scores'][j], category_index,
                                                               instance_masks=D.get('detection_masks'),
                                                               use_normalized_coordinates=True, line_thickness=8)
            out.write(IM[j])  # wants BGR
            # plots.imshow(im)
    else:
        for i in range(nframes):
            success, im = cap.read()  # native BGR
            if success:
                tic = time.time()
                im = cv2.resize(im, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                D = sess.run(T, feed_dict={image_tensor: np.expand_dims(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), 0)})
                dt[i] = time.time() - tic
                vis_util.visualize_boxes_and_labels_on_image_array(im, D['detection_boxes'][0],
                                                                   D['detection_classes'][0].astype(np.uint8),
                                                                   D['detection_scores'][0], category_index,
                                                                   use_normalized_coordinates=True, line_thickness=8)
                s = '%s %5.3fs (mean %5.3fs)' % (mname, dt[i], dt[min(i, 1):i + 1].mean())
                print('%g/%g %s' % (i, nframes, s))
                im = annotateImageSpeed(im, s)
                out.write(im)  # wants BGR
            else:
                break

    print('All done... %.3fs.' % (time.time() - tic0))
    cap.release()
    out.release()
    sess.close()


mainYOLO()
# MODEL_NAME, mname = 'data/ssd_mobilenet_v1_coco_2017_11_17', 'ssd_mobilenetv1linear_'
# mainTF(MODEL_NAME, mname)
# MODEL_NAME, mname = 'data/ssd_mobilenet_v2_coco_2018_03_29', 'ssd_mobilenetv2linear_'
# mainTF(MODEL_NAME, mname)
