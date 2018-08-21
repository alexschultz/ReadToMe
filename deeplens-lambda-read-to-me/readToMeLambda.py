import mo
import os
import traceback
from threading import Thread, Timer
import logging
from time import sleep
import sys
import awscam
import cv2
import greengrasssdk
import numpy as np
import speak
import time
from boto3 import client
import imageProcessing as ip

logger = logging.getLogger()
logger.setLevel(logging.INFO)
client = greengrasssdk.client('iot-data')
# The information exchanged between IoT and CLOUD has
# a topic and a message body.
# This is the topic that this code uses to send messages to cloud
iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
ret, frame = awscam.getLastFrame()
Write_To_FIFO = True
FIRST_RUN = True
PLAY_INTRO = False
input_width = 300
input_height = 300
prob_thresh = 0.55
outMap = {1: 'text_block'}
model_name = "read-to-me"
occursThreshold = 5 
error, model_path = mo.optimize(model_name, input_width, input_height)
model = awscam.Model(model_path, {"GPU": 1})
client.publish(topic=iot_topic, payload="Model loaded.")
model_type = "ssd"
jpeg = None

def firstRunFunc():
    logger.info('first run')
    global FIRST_RUN
    try:
        if PLAY_INTRO:
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
                if jpeg is not None:
		    f.write(jpeg.tobytes())
            except IOError as e:
                continue


def greengrass_infinite_infer_run():
    logger.info('first run {}'.format(FIRST_RUN))
    if FIRST_RUN:
        firstRunFunc()
    try:
        results_thread = FIFO_Thread()
        results_thread.start()
        client.publish(topic=iot_topic, payload="Inference starting")
        ret, frame = awscam.getLastFrame()
        if ret == False:
            raise Exception("Failed to get frame from the stream")
        yscale = float(frame.shape[0] / input_height)
        xscale = float(frame.shape[1] / input_width)
        occurs = 0
        doInfer = True
        while doInfer:
            frameContainsText = False
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
            label = ''
            text_blocks = filter(lambda x: x['prob'] > prob_thresh, parsed_results)
            # see the order of the text blocks
            # for this, we really only want to grab the highest probable text block, so just ignore the rest
            if len(text_blocks) > 0:
                obj = text_blocks[0]
                logger.debug('number of text blocks detected: [{}]'.format(len(text_blocks)))
		text_blocks.sort(key=lambda x: x['prob'])
                text_blocks = text_blocks[len(text_blocks) - 1:]
                frameContainsText = True
                occurs += 1
                logger.debug('text blocks detected')
                xmin = int(xscale * obj['xmin']) + int((obj['xmin'] - input_width / 2) + input_width / 2)
                ymin = int(yscale * obj['ymin'])
                xmax = int(xscale * obj['xmax']) + int((obj['xmax'] - input_width / 2) + input_width / 2)
                ymax = int(yscale * obj['ymax'])
                logger.debug('xmin {} xmax {} ymin {} ymax {}'.format(xmin, xmax, ymin, ymax))
                label_show = "{}: conseq: {}:    {:.2f}%".format(outMap[obj['label']], occurs, obj['prob'] * 100)
                if occurs >= (occursThreshold - 1):
                    try:
                        occurs = 0
                        logger.debug('ocr iomage')
                        tb = ip.getRoi(frame, xmin, xmax, ymin, ymax)
                        tb = ip.correctSkew(tb)
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (121, 255, 20), 4)
                        cv2.putText(frame, label_show, (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (121, 255, 20), 4)
                        clean = ip.cleanUpTextArea(tb)
                        txt = ip.ocrImage(clean, extractBadChars=True, spellCheck=False)
                        if txt == '':
                            speak.speak('Sorry, I am unable to read the page. \nPlease try again.')
                        else:
                            logger.info(txt)
                            speak.speak(txt)

                    except Exception as e:
                        msg = "ocr failed: " + str(e)
                        cv2.imwrite(os.path.join(os.path.abspath(os.sep),'tmp', '{}-{}-{}-{}-{}.jpg'.format(time.strftime("%Y-%m-%d %H:%M:%S"),xmin, xmax, ymin, ymax)), tb)
                        client.publish(topic=iot_topic, payload=msg)

                else:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 165, 20), 4)
                    cv2.putText(frame, label_show, (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 20), 4)
                
            else:
                occurs = 0

            global jpeg
            shp = frame.shape
            smallerFrame = cv2.resize(frame, (shp[1]/3, shp[0]/3))          
            ret, jpeg = cv2.imencode('.jpg', smallerFrame) 
            #ret, jpeg = cv2.imencode('.jpg', frame)

    except Exception as e:
        msg = "Lambda function failed: " + str(e)
        logger.info(msg)
        speak.speak("I'm sorry, I wasn't able to read that for some reason.")

    # Asynchronously schedule this function to be run again in 15 seconds
    Timer(15, greengrass_infinite_infer_run).start()


# run the function and view the results
greengrass_infinite_infer_run()


def function_handler(event, context):
    return

