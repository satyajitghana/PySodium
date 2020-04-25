# USAGE
# python yolo.py --image images/baggage_claim.jpg --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os
from tqdm.auto import tqdm
from sodium.utils import setup_logger

logger = setup_logger(__name__)

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                 help="path to input image")
# ap.add_argument("-y", "--yolo", required=True,
#                 help="base path to YOLO directory")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
#                 help="minimum probability to filter weak detections")
# ap.add_argument("-t", "--threshold", type=float, default=0.3,
#                 help="threshold when applyong non-maxima suppression")
# args = vars(ap.parse_args())


def gen_bar_updater():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """
    import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    os.makedirs(root, exist_ok=True)

    # check if file is already present locally
    if os.path.exists(os.path.join(root, filename)):
        print('Using downloaded file: ' + fpath)
    else:   # download the file
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater()
            )
        except (urllib.error.URLError, IOError) as e:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater()
                )
            else:
                raise e


class YoloOpenCV:
    YOLO_COCO_DIR = 'yolo-coco'

    YOLO_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    WEIGHTS_URL = 'https://pjreddie.com/media/files/yolov3.weights'
    WEIGHTS_FILENAME = 'yolov3.weights'

    def __init__(self):
        # download the weights
        self.download_weights()

        # load the COCO class labels our YOLO model was trained on
        labelsPath = os.path.join(
            self.YOLO_DIR, self.YOLO_COCO_DIR, "coco.names")
        self.LABELS = open(labelsPath).read().strip().split("\n")

        # initialize a list of colors to represent each possible class label
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
                                        dtype="uint8")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.join(
            self.YOLO_DIR, self.YOLO_COCO_DIR, "yolov3.weights")
        configPath = os.path.join(
            self.YOLO_DIR, self.YOLO_COCO_DIR, "yolov3.cfg")

        # load our YOLO object detector trained on COCO dataset (80 classes)
        logger.info("loading YOLO model")
        self.YoloV3 = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    def download_weights(self):
        logger.info('Downloading Weights')
        download_url(self.WEIGHTS_URL, os.path.join(
            self.YOLO_DIR, self.YOLO_COCO_DIR), self.WEIGHTS_FILENAME)

    def run_yolo_on_image(self, image_path, confidence_min=0.5, threshold=0.3):

        # load our input image and grab its spatial dimensions
        image = cv2.imread(image_path)
        (H, W) = image.shape[:2]

        # determine only the *output* layer names that we need from YOLO
        ln = self.YoloV3.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.YoloV3.getUnconnectedOutLayers()]

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        self.YoloV3.setInput(blob)
        start = time.time()
        layerOutputs = self.YoloV3.forward(ln)
        end = time.time()

        # show timing information on YOLO
        logger.info("YOLO took {:.6f} seconds".format(end - start))

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > confidence_min:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_min, threshold)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(
                    self.LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
