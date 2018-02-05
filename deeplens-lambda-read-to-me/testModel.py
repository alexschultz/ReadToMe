import cv2
import collections
import os
import tensorflow as tf
from object_detection.utils import label_map_util
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import numpy as np
import speak
import pytesseract
from string import ascii_letters, digits

cap = cv2.VideoCapture(0)
MODEL_NAME = 'tensorflow-model'
prob_thresh = 0.65
tbFrameCount = 0
tbFramThresh = 10
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
NUM_CLASSES = 1
# List of the strings that is used to add correct label for each box.
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

speak.playAudioFile(os.path.join('staticfiles', 'intro.mp3'))
speak.playAudioFile(os.path.join('staticfiles', 'dir1.mp3'))
speak.playAudioFile(os.path.join('staticfiles', 'chime.mp3'))
speak.playAudioFile(os.path.join('staticfiles', 'dir2.mp3'))
speak.playAudioFile(os.path.join('staticfiles', 'dir3.mp3'))
speak.playAudioFile(os.path.join('staticfiles', 'dir4.mp3'))
speak.playAudioFile(os.path.join('staticfiles', 'chime.mp3'))

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


def ExtractAlphanumeric(InputString):
    """Remove junk characters from OCR text output.
    Tesseract is pretty good, but sometimes it spits out a bunch of garbage characters
    So this function strips out any non alpha numeric characteers as well as normal punctuation marks
    before sending it off to AWS Polly to be turned into audio
    """
    line = InputString.replace("\n", " ")
    return "".join([ch for ch in line if ch in (ascii_letters + digits + " " + "-" + "!" + '?')])


def ocrImage(pilImage):
    text = pytesseract.image_to_string(pilImage, lang="eng")
    return ExtractAlphanumeric(text)


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        tbFrameCount = 0
        foundTextBlock = False
        while True:
            foundTextBlock = False
            ret, image_np = cap.read()
            image_np = cv2.resize(image_np, (1024, 768))
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            box_to_color_map = collections.defaultdict(str)
            box_to_display_str_map = collections.defaultdict(list)

            bxs = np.squeeze(boxes)
            cls = np.squeeze(classes).astype(np.int32)
            srs = np.squeeze(scores)

            for i in range(min(1, bxs.shape[0])):
                if srs is None or srs[i] > prob_thresh:
                    # We found something in this frame, set foundTextBlock to true and increment the foundFrameCount
                    foundTextBlock = True
                    tbFrameCount = tbFrameCount + 1
                    box = tuple(bxs[i].tolist())
                    if cls[i] in category_index.keys():
                        class_name = category_index[cls[i]]['name']
                    else:
                        class_name = 'N/A'
                    display_str = '{}: {}%'.format(class_name, int(100 * srs[i]))
                    # print(display_str)
                    box_to_display_str_map[box].append(display_str)
                    box_to_color_map[box] = 'AliceBlue'

                    for box, color in box_to_color_map.items():
                        ymin, xmin, ymax, xmax = box
                        image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
                        draw = ImageDraw.Draw(image_pil)
                        im_width, im_height = image_pil.size
                        # We add 30 to all the coordinates to make sure we get all the text inside the box
                        (left, right, top, bottom) = (xmin * im_width - 30, xmax * im_width + 30,
                                                      ymin * im_height - 30, ymax * im_height + 30)

                        draw.line([(left, top), (left, bottom), (right, bottom),
                                   (right, top), (left, top)], width=1, fill=color)

                        try:
                            tbPilImage = Image.fromarray(np.uint8(image_np)).convert('RGB')
                            tbImage = tbPilImage.crop((left, top, right, bottom))
                            # Show every 3rd frame
                            if tbFrameCount % 2 == 0:
                                textAreaImage = cleanUpTextArea(np.array(tbImage))
                                cv2.imshow('text area', textAreaImage)
                            if tbFrameCount >= tbFramThresh:
                                speak.playAudioFile(os.path.join('staticfiles', 'chime.mp3'))
                                text = ocrImage(Image.fromarray(np.uint8(textAreaImage)))
                                print(text)
                                speak.speak(text)
                                tbFrameCount = 0

                        except IOError:
                            print("error grabbing text block")
                        try:
                            font = ImageFont.truetype('arial.ttf', 24)
                        except IOError:
                            font = ImageFont.load_default()

                        np.copyto(image_np, np.array(image_pil))

            cv2.imshow('object detection', image_np)
            # If we didn't find a text block in the last frame, reset the counter
            # this should help with reducing motion blur
            if not foundTextBlock:
                tbFrameCount = 0

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

cap.release()
cv2.destroyAllWindows()
