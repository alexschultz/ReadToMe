import logging
import os
import sys
import traceback
from threading import Thread
from threading import Timer
from time import sleep
import tensorflow as tf
from object_detection.utils import label_map_util
import awscam
import cv2
import numpy as np
import greengrasssdk
from boto3 import client
import speak

logger = logging.getLogger()
logger.setLevel(logging.INFO)

client = greengrasssdk.client('iot-data')

# The information exchanged between IoT and clould has
# a topic and a message body.
# This is the topic that this code uses to send messages to cloud
iotTopic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
ret, frame = awscam.getLastFrame()
ret, jpeg = cv2.imencode('.jpg', frame)
Write_To_FIFO = True
FIRST_RUN = True

MODEL_NAME = 'tensorflow-model'
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
NUM_CLASSES = 1
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'object-detection.pbtxt')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def firstRunFunc():
    logger.info('first run')
    global FIRST_RUN
    try:
        speak.playAudioFile("staticfiles/greeting.mp3")
        sleep(0.5)
        speak.playAudioFile("staticfiles/instructions.mp3")
        sleep(1)
        speak.playAudioFile("staticfiles/get-started.mp3")
        FIRST_RUN = False
    except:
        print("exception occurred!")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5, file=sys.stdout)


def ocrTest():
    return "getting text from tesseract"

class FIFO_Thread(Thread):
    def __init__(self):
        ''' Constructor. '''
        Thread.__init__(self)

    def run(self):
        fifo_path = "/tmp/results.mjpeg"
        if not os.path.exists(fifo_path):
            os.mkfifo(fifo_path)
        f = open(fifo_path, 'w')
        client.publish(topic=iotTopic, payload="Opened Pipe")
        while Write_To_FIFO:
            try:
                f.write(jpeg.tobytes())
            except IOError as e:
                continue


def greengrass_infinite_infer_run():
    logger.info('starting lambda')
    global FIRST_RUN
    logger.info('first run {}'.format(FIRST_RUN))
    if FIRST_RUN:
        firstRunFunc()
    try:
        input_width = 300
        input_height = 300
        prob_thresh = 0.65
        results_thread = FIFO_Thread()
        results_thread.start()

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # Send a starting message to IoT console
        client.publish(topic=iotTopic, payload="Text detection starts now")

        ret, frame = awscam.getLastFrame()
        if ret == False:
            raise Exception("Failed to get frame from the stream")

        yscale = float(frame.shape[0] / input_height)
        xscale = float(frame.shape[1] / input_width)
        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                while True:
                    # Get a frame from the video stream
                    ret, frame = awscam.getLastFrame()
                    # Raise an exception if failing to get a frame
                    if ret == False:
                        raise Exception("Failed to get frame from the stream")

                    # Resize frame to fit model input requirement
                    image_np_expanded = np.expand_dims(frame, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                  # TODO see test script to get this working
                    print("boxes: {}".format(len(boxes)))

            global jpeg
            ret, jpeg = cv2.imencode('.jpg', frame)

    except Exception as e:
        msg = "OCR failed: " + str(e)
        speak.talk("I'm sorry, I wasn't able to read that for some reason.")
        client.publish(topic=iotTopic, payload=msg)

    # Asynchronously schedule this function to be run again in 15 seconds
    Timer(15, greengrass_infinite_infer_run).start()


# Execute the function above
greengrass_infinite_infer_run()


def function_handler(event, context):
    return
