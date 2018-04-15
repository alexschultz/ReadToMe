import mo
import os
import traceback
from string import ascii_letters, digits
from threading import Thread, Timer
import logging
from time import sleep
import sys
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import awscam
import cv2
import greengrasssdk
import numpy as np
import pytesseract
import speak
from boto3 import client

logger = logging.getLogger()
logger.setLevel(logging.INFO)

client = greengrasssdk.client('iot-data')

# The information exchanged between IoT and CLOUD has
# a topic and a message body.
# This is the topic that this code uses to send messages to cloud
iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])

ret, frame = awscam.getLastFrame()
jpeg = None
Write_To_FIFO = True
FIRST_RUN = True


def firstRunFunc():
    logger.info('first run')
    global FIRST_RUN
    try:
        speak.playAudioFile(os.path.join('staticfiles', 'intro.mp3'))
        sleep(0.5)
        speak.playAudioFile(os.path.join('staticfiles', 'dir1.mp3'))
        speak.playAudioFile(os.path.join('staticfiles', 'chime.mp3'))
        speak.playAudioFile(os.path.join('staticfiles', 'dir2.mp3'))
        sleep(1)
        speak.playAudioFile(os.path.join('staticfiles', 'dir3.mp3'))
        sleep(.5)
        speak.playAudioFile(os.path.join('staticfiles', 'dir4.mp3'))
        sleep(0.5)
        speak.playAudioFile(os.path.join('staticfiles', 'chime.mp3'))
        FIRST_RUN = False
    except:
        print("exception occurred!")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5, file=sys.stdout)


def cleanUpTextArea(image):
    try:
        height, width = image.shape[:2]
        res = cv2.resize(image, (3 * width, 3 * height), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        denoise = cv2.fastNlMeansDenoising(blur)
        thresh = cv2.adaptiveThreshold(denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        blur1 = cv2.GaussianBlur(thresh, (5, 5), 0)
        final = cv2.GaussianBlur(blur1, (5, 5), 0)
        return final
    except Exception:
        print("unable to display text block")


def ocrImage(pilImage):
    text = pytesseract.image_to_string(pilImage, lang="eng")
    return ExtractAlphanumeric(text)


def ExtractAlphanumeric(InputString):
    """Remove junk characters from OCR text output.
    Tesseract is pretty good, but sometimes it spits out a bunch of garbage characters
    So this function strips out any non alpha numeric characteers as well as normal punctuation marks
    before sending it off to AWS Polly to be turned into audio
    """
    line = InputString.replace("\n", " ")
    return "".join([ch for ch in line if ch in (ascii_letters + digits + " " + "-" + "!" + '?')])


def ocrTest():
    return "getting text from tesseract"


# create a simple class that runs on its own thread so we can publish output images
#    to the FIFO file and view using mplayer
class FIFO_Thread(Thread):
    def __init__(self):
        '''Constructor.'''
        Thread.__init__(self)

    def run(self):
        fifo_path = "/tmp/results.mjpeg"
        if not os.path.exists(fifo_path):
            os.mkfifo(fifo_path)
        f = open(fifo_path, 'w')
        client.publish(topic=iot_topic, payload="Opened Pipe")

        while Write_To_FIFO:
            try:
                f.write(jpeg.tobytes())
            except IOError as e:
                continue


def greengrass_infinite_infer_run():
    input_width = 300
    input_height = 300
    model_name = "read-to-me"

    error, model_path = mo.optimize(model_name, input_width, input_height)

    model = awscam.Model(model_path, {"GPU": 1})
    client.publish(topic=iot_topic, payload="Model loaded.")

    model_type = "ssd"

    # load the labels into a list where the index represents the label returned by the network
    with open('labels.txt', 'r') as f:
        labels = [l.rstrip() for l in f]

    # define the number of classifiers to see

    jpeg = None
    Write_To_FIFO = True
    logger.info('starting lambda')
    global FIRST_RUN
    logger.info('first run {}'.format(FIRST_RUN))
    if FIRST_RUN:
        firstRunFunc()
    try:
        input_width = 300
        input_height = 300
        prob_thresh = 0.55
        topk = 2
        # start the FIFO thread to view the output locally
        results_thread = FIFO_Thread()
        results_thread.start()
        # you can publish an "Inference starting" message to the AWS IoT console
        client.publish(topic=iot_topic, payload="Inference starting")
        # access the latest frame on the mjpeg stream
        ret, frame = awscam.getLastFrame()
        if ret == False:
            raise Exception("Failed to get frame from the stream")

        doInfer = True
        while doInfer:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            # Raise an exception if failing to get a frame
            if ret == False:
                raise Exception("Failed to get frame from the stream")

            # Resize frame to fit model input requirement
            frameResize = cv2.resize(frame, (input_width, input_height))

            # Run model inference on the resized frame
            inferOutput = model.doInference(frameResize)

            # Output inference result to the fifo file so it can be viewed with mplayer
            parsed_results = model.parseResult(model_type, inferOutput)[model_type]
            label = '{'
            frameContainsText = False
            for obj in parsed_results:
                if obj['prob'] < prob_thresh:
                    break
                else:
                    frameContainsText = True

            label += '"null": 0.0'
            label += '}'
            client.publish(topic=iot_topic, payload=label)
            if frameContainsText:
                try:
                    # tesResults = get_text_from_cv2_image(frame)
                    client.publish(topic=iot_topic, payload='{{"output":"{}"}}'.format('found text_block'))
                except Exception as e:
                    msg = "ocr failed: " + str(e)
                    client.publish(topic=iot_topic, payload=msg)

            global jpeg
            ret, jpeg = cv2.imencode('.jpg', frame)
    except Exception as e:
        msg = "OCR failed: " + str(e)
        speak.speak("I'm sorry, I wasn't able to read that for some reason.")
        msg = "Lambda function failed: " + str(e)
        client.publish(topic=iot_topic, payload=msg)

    # Asynchronously schedule this function to be run again in 15 seconds
    Timer(15, greengrass_infinite_infer_run).start()


# run the function and view the results
greengrass_infinite_infer_run()


def function_handler(event, context):
    return
