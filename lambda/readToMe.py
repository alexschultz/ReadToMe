from threading import Thread, Timer
from time import sleep
import mo
import awscam
import os
import sys
import imageProcessing as ip
import traceback
import speak
import time
import logging
import greengrasssdk
import cv2


Write_To_FIFO = True
FIRST_RUN = True
PLAY_INTRO = True
input_width = 300
input_height = 300
prob_thresh = 0.55
outMap = {0: 'text_block'}
model_name = "read-to-me"
occursThreshold = 10
client = greengrasssdk.client('iot-data')
iot_topic = '$aws/things/{}/infer'.format(os.environ['AWS_IOT_THING_NAME'])
jpeg = None
logger = logging.getLogger()
logger.setLevel(logging.INFO)
error, model_path = mo.optimize(model_name, input_width, input_height)
model = awscam.Model(model_path, {"GPU": 1}, awscam.Runtime.DLDT)

def log_message(message):
    logger.info(message)
#    client.publish(topic=iot_topic, payload=message)

def first_run():
    log_message('first run')
    global FIRST_RUN
    try:
        if PLAY_INTRO:
            log_message('attempting to play audio files for first run')
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
        log_message("exception occurred while trying to play audio files")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=5, file=sys.stdout)


# create a simple class that runs on its own thread so we can publish output images
# to the FIFO file and view using mplayer
class FIFO_Thread(Thread):
    def __init__(self):
        '''Constructor.'''
        Thread.__init__(self)

    def run(self):
        fifo_path = "/tmp/results.mjpeg"
        if not os.path.exists(fifo_path):
            os.mkfifo(fifo_path)

        f = open(fifo_path, 'w')
        log_message('Opened Pipe')

        while Write_To_FIFO:
            try:
                if jpeg is not None:
                    f.write(jpeg.tobytes())
            except IOError as e:
                continue


def greengrass_infinite_infer_run():
    if FIRST_RUN:
        log_message('first run {}'.format(FIRST_RUN))
        first_run()
    try:
        results_thread = FIFO_Thread()
        results_thread.start()
        ret, frame = awscam.getLastFrame()
        log_message('Inference starting...')
        if not ret:
            log_message('Failed to get frame from the stream')
            raise Exception('Failed to get frame from the stream')

        yscale = float(frame.shape[0] / input_height)
        xscale = float(frame.shape[1] / input_width)
        occurs = 0
        while True:
            # Get a frame from the video stream
            ret, frame = awscam.getLastFrame()
            # Raise an exception if failing to get a frame
            if not ret:
                raise Exception("Failed to get frame from the stream")

            # Resize frame to fit model input requirement
            resized_frame = cv2.resize(frame, (input_width, input_height))
            # Run model inference on the resized frame
            # log_message('debugging before inference')
            output = model.doInference(resized_frame)
            # log_message('debugging after inference')
            results = model.parseResult('ssd', output)['ssd']
            # log_message('debugging after parseResult')
            text_blocks = filter(lambda x: x['prob'] > prob_thresh, results)
            # see the order of the text blocks
            # for this, we really only want to grab the highest probable text block, so just ignore the rest
            if len(text_blocks) > 0:
                obj = text_blocks[0]
                log_message('number of text blocks detected: [{}]'.format(len(text_blocks)))
                text_blocks.sort(key=lambda x: x['prob'])
                text_blocks = text_blocks[len(text_blocks) - 1:]
                occurs += 1
                log_message('text blocks detected')
                xmin = int(xscale * obj['xmin']) + int((obj['xmin'] - input_width / 2) + input_width / 2)
                ymin = int(yscale * obj['ymin'])
                xmax = int(xscale * obj['xmax']) + int((obj['xmax'] - input_width / 2) + input_width / 2)
                ymax = int(yscale * obj['ymax'])
                log_message('xmin {} xmax {} ymin {} ymax {}'.format(xmin, xmax, ymin, ymax))
                log_message(obj)
                label_show = "{}: conseq: {}:    {:.2f}%".format(outMap[obj['label']], occurs, obj['prob'] * 100)
                log_message(label_show)
                if occurs >= occursThreshold:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (121, 255, 20), 4)
                    cv2.putText(frame, label_show, (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (121, 255, 20), 4)
                    speak.playAudioFile(os.path.join('staticfiles', 'chime.mp3'))
                    try:
                        occurs = 0
                        log_message('ocr image')
                        tb = ip.getRoi(frame, xmin, xmax, ymin, ymax)
                        tb = ip.correctSkew(tb)
                        clean = ip.cleanUpTextArea(tb)
                        txt = ip.ocrImage(clean, extractBadChars=True, spellCheck=False)
                        if txt == '':
                            log_message('ocr returned received nothing for text')
                            speak.speak('Sorry, I am unable to read the page. \nPlease try again.')
                        else:
                            log_message(txt)
                            speak.speak(txt)

                    except Exception as e:
                        msg = "ocr failed: " + str(e)
                        cv2.imwrite(os.path.join(os.path.abspath(os.sep), 'tmp',
                                                 '{}-{}-{}-{}-{}.jpg'.format(time.strftime("%Y-%m-%d %H:%M:%S"), xmin,
                                                                             xmax, ymin, ymax)), tb)
                        log_message(msg)

                else:
                    log_message('showing bbox after {}'.format(occurs))
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 165, 20), 4)
                    log_message('showed bbox')
                    cv2.putText(frame, label_show, (xmin, ymin - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 20), 4)

            else:
                occurs = 0

            global jpeg
            shp = frame.shape
            smaller_frame = cv2.resize(frame, (shp[1] / 3, shp[0] / 3))
            ret, jpeg = cv2.imencode('.jpg', smaller_frame)

    except Exception as e:
        msg = "Lambda function failed: " + str(e)
        log_message(msg)
        speak.speak("I'm sorry, I wasn't able to read that for some reason.")

    # Asynchronously schedule this function to be run again in 15 seconds
    Timer(3, greengrass_infinite_infer_run).start()


# run the function and view the results
greengrass_infinite_infer_run()

def function_handler(event, context):
    return
