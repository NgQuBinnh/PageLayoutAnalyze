"""Evaluation script for the DeepLab-ResNet network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on 1449 validation images.
"""

from __future__ import print_function

import argparse
import json
import collections
from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, prepare_label
from PIL import Image
from matplotlib import pyplot as plt
from os.path import join as osj
import tensorflow as tf
import numpy as np
from read_anntations import *
import cv2
from deeplab_resnet import DeepLabResNetModel, ImageReader, prepare_label

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = 'D:\Data\PLAD'
DATA_LIST_PATH = './dataset/test_plad.txt'
IGNORE_LABEL = 255
NUM_CLASSES = 4
NUM_STEPS = 35 # Number of images in the validation set.
OUT_DIR = ""
# RESTORE_FROM = './deeplab_resnet.ckpt'
RESTORE_FROM = 'D:\Coding\page_layout_analysis\snapshots\model.ckpt-430'
IMAGE = ""
IMAGE_PATH = ""
IS_POD = False


def bfs_in_image(matrix):
    area = -1
    visited = set()
    dx = [-1, 0, 1, 0]
    dy = [0, 1, 0, -1]
    list_bbxs = []
    for x, row in enumerate(matrix):
        for y, col in enumerate(row):
            if matrix[x][y] == 1 and (x, y) not in visited:

                area += 1
                minx, miny, maxx, maxy = x, y, x, y
                queue = collections.deque([(x, y)])
                while queue:
                    (x, y) = queue.popleft()
                    for i in range(4):
                        nx = x + dx[i]
                        ny = y + dy[i]
                        if (nx > -1) and (ny > -1)\
                                and (nx < matrix.shape[0]) and (ny < matrix.shape[1])\
                                and (nx, ny) not in visited and matrix[nx][ny] == 1:
                            visited.add((nx, ny))
                            queue.append((nx, ny))
                            minx = min(minx, nx)
                            miny = min(miny, ny)
                            maxx = max(maxx, nx)
                            maxy = max(maxy, ny)
                list_bbxs.append((minx, miny, maxx, maxy))

    return list_bbxs


def thresholding(matrix, threshold=0.60):
    result = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=int)
    for x, row in enumerate(matrix):
        for y, col in enumerate(row):
            if col > threshold:
                result[x][y] = 1
    return result


def drw_img(img, lbs, bbxs):
    for idx, bbx in enumerate(bbxs):
        if lbs[idx] == "formula":
            cv2.rectangle(img, (bbx[1], bbx[0]), (bbx[3], bbx[2]), (128,0,0) , 3)
        else:
            if lbs[idx] == "figure":
                cv2.rectangle(img, (bbx[1], bbx[0]), (bbx[3], bbx[2]), (0,128,0), 3)
            else:
                if lbs[idx] == "table":
                    cv2.rectangle(img, (bbx[1], bbx[0]), (bbx[3], bbx[2]), (128, 128, 0), 3)
    cv2.imwrite("bounding_box_output/" + IMAGE + ".png", img)


def evaluate_confidence(img, bbx):
    total = 0.0
    nb = 0
    for x, row in enumerate(img):
        for y, col in enumerate(row):
            if (x >= bbx[0]) and (x <= bbx[2]) and (y >= bbx[1]) and (y <= bbx[3]):
                nb += 1
                total += col
    return total/nb


def in_bbox(ba, bb):
    if bb[0] <= ba[0] and ba[2] <= bb[2] and bb[1] <= ba[1] and ba[3] <= bb[3]:
        return True
    return False


def get_bounding_boxs(array):
    labels = []
    bounding_boxs = []
    confidence_score = []
    pixel_score = 0.0
    miuo_score = 0

    txtpath = osj('D:\Data\POD\Annotations', IMAGE + '.xml')
    data = FileData()
    data.readtxt(txtpath)
    ts = 0.0
    ms = 0.0
    for label in range(1, 4, 1):
        type_label = ["", "formula", "figure", "table"]

        probabilities_np = array.squeeze()[:, :, label]
        thres_mt = thresholding(probabilities_np)
        bboxs = bfs_in_image(thres_mt)
        true_pred, total_pixel_class = pixel_acc(thres_mt, type_label[label], data.pageLines)
        ts += true_pred
        ms += total_pixel_class
        # cmap = plt.get_cmap('bwr')
        # f, (ax1) = plt.subplots(1, 1, sharey=True)
        #
        # probability_graph_0 = ax1.imshow(array.squeeze()[:, :, label])
        # ax1.set_title("Probability map of " + type_label[label])
        #
        # plt.colorbar(probability_graph_0)
        # plt.show()

        min_area = [0, 590, 1500, 1500]

        for idx, bbox in enumerate(bboxs):
            if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) <= min_area[label]:
                continue
            labels.append(type_label[label])
            bounding_boxs.append(bbox)
            confidence_score.append(evaluate_confidence(probabilities_np, bbox))
    pixel_score = (ts/ms)

    indices = [1 for x in range(len(bounding_boxs))]
    for idx, bbox in enumerate(bounding_boxs):
        for i in range(idx + 1, len(bounding_boxs), 1):
            if in_bbox(bounding_boxs[idx], bounding_boxs[i]):
                indices[idx] = 0
            else:
                if in_bbox(bounding_boxs[i], bounding_boxs[idx]):
                    indices[i] = 0

    for idx in range(len(bounding_boxs) - 1, -1, -1):
        if indices[idx] == 0:
            labels.pop(idx)
            bounding_boxs.pop(idx)
            confidence_score.pop(idx)

    miuo_score = miuo(np.zeros((array.shape[1], array.shape[2])),
                      bounding_boxs,
                      data.pageLines)

    return labels, bounding_boxs, confidence_score, miuo_score, pixel_score


def pixel_acc(matrix, label, page_lines):
    TP = TN = 0.0
    union = 0.0
    area = 0.0
    for page_line in page_lines:
        # print(label + " ? " + page_line.kind)
        if page_line.kind == label:
            bbx = page_line.rect
            area += (bbx.r - bbx.l + 1) * (bbx.d - bbx.u + 1)
            for x in range(bbx.l, bbx.r + 1, 1):
                for y in range(bbx.u, bbx.d + 1, 1):
                    matrix[y][x] += 1

    for x, row in enumerate(matrix):
        for y, value in enumerate(row):
            TP += (value == 2)
            TN += (value == 0)
            union += (value == 2 or value == 1)

    return TP, area


def pixel_acc_mask(predict, label, mskp):
    total = 0.0
    true_predict = 0.0
    for x, row in enumerate(predict):
        for y, value in enumerate(row):
            if mskp[x][y] == label:
                total += 1
            if value == 1 and label == mskp[x][y]:
                true_predict += 1
    return true_predict, total


def probabilities_map_evaluate(array, mskp):
    ts = ms = 0.0

    for label in range(1, 4, 1):
        type_label = ['', 'TableRegion', 'FrameRegion', 'TextRegion']

        probabilities_np = array.squeeze()[:, :, label]
        thres_mt = thresholding(probabilities_np)
        ground_truth = cv2.imread(mskp, cv2.IMREAD_GRAYSCALE)
        true_pred, total_pixel_class = pixel_acc_mask(thres_mt, label, ground_truth)
        ts += true_pred
        ms += total_pixel_class

    return ts / ms


def miuo(matrix, predicted, ground_truth):
    overlap = 0.0
    union = 0.0

    for bbx in predicted:
        for x in range(bbx[0], bbx[2] + 1, 1):
            for y in range(bbx[1], bbx[3] + 1, 1):
                matrix[x][y] += 1

    for page_line in ground_truth:
        bbx = page_line.rect

        for x in range(bbx.l, bbx.r + 1, 1):
            for y in range(bbx.u, bbx.d + 1, 1):
                matrix[y][x] += 1

    for x, row in enumerate(matrix):
        for y, value in enumerate(row):
            overlap += (value == 2)
            union += (value == 2 or value == 1)

    return overlap/union

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load validation

    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            None, # No defined input size.
            False, # No random scale.
            False, # No random mirror.
            args.ignore_label,
            IMG_MEAN,
            coord)
        image, label, file, mask = reader.image, reader.label, reader.image_list, reader.label_list
    image_batch, label_batch, file_batch, mask_batch = tf.expand_dims(image, dim=0), \
                                                       tf.expand_dims(label, dim=0), \
                                                       tf.expand_dims(file, dim=0), \
                                                       tf.expand_dims(mask, dim=0)# Add one batch dimension.

    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, is_training=False, num_classes=args.num_classes)

    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    probabilities = tf.nn.softmax(raw_output)

    raw_output = tf.argmax(raw_output, dimension=3)
    preds = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.
    
    # mIoU
    # pred = tf.reshape(preds, [-1,])
    # gt = tf.reshape(label_batch, [-1,])
    # weights = tf.cast(tf.less_equal(gt, args.num_classes - 1), tf.int32) # Ignoring all labels greater than or equal to n_classes.
    # mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=args.num_classes, weights=weights)
    file_name = file_batch
    mask_link = mask_batch
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if args.restore_from is not None:
        load(loader, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    # Iterate over training steps.
    result_data = {}

    total_miuo_score = 0
    nb = 0
    global OUT_DIR
    global IMAGE
    global IMAGE_PATH

    for step in range(args.num_steps):
        predict, probmap, fb, mk = sess.run([preds, probabilities, file_name, mask_link])
        print('step {:d}'.format(step))

        if IS_POD:
            IMAGE = fb[0][step].decode('utf8').replace("D:\Data\POD/JpegImages/", "").replace(".jpg", "")
            IMAGE_PATH = "D:\Data\POD\JpegImages\\" + IMAGE + ".jpg"

            labels, bounding_boxs, confidence_score, miuo_score, pixel_score = \
                get_bounding_boxs(probmap)

            result_data[IMAGE] = {
                "labels": labels,
                "confidence_score": confidence_score,
                "boxes": bounding_boxs,
                "pixel_accuracy": pixel_score,
                "mIoU": miuo_score
            }

            # Draw bouding boxes to image
            # drw_img(cv2.imread(IMAGE_PATH), labels, bounding_boxs)

            OUT_DIR = "mask_pod/"
        else:
            # print(mk[0][step].decode('utf8'))
            IMAGE = mk[0][step].decode('utf8').replace("D:\Data\PLAD/MaskImage/", "")
            result_data[IMAGE] = probabilities_map_evaluate(probmap, mk[0][step].decode('utf8'))
            OUT_DIR = "mask_plad/"

        print(result_data)

        # Draw mask image
        msk = decode_labels(predict, num_classes=args.num_classes)
        im = Image.fromarray(msk[0])
        im.save( OUT_DIR + IMAGE + '.png')

        # with open('predicted_boxes.json', 'w') as outfile:
        #     json.dump(result_data, outfile)

    # print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
