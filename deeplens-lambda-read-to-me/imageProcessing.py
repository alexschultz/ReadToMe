import cv2
import numpy as np
import pytesseract
from autocorrect import spell
from string import printable

def getRoi(image, x1, x2, y1, y2):
    ht, wd, ch = image.shape
    print('height {} width {} channels {}'.format(ht, wd, ch))
    #xmin = x1
    #xmax = x2
    #ymin = y1
    #ymax = y2
    xmin = x1 - 12
    xmax = x2 + 12
    ymin = y1 - 12
    ymax = y2 + 12

    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0
    if xmax > wd:
        xmax = wd
    if ymax > ht:
        ymax = ht
    print('xmin {} xmax {} ymin {} ymax {}'.format(xmin, xmax, ymin, ymax))
    tb = image[ymin:ymax, xmin:xmax]
    return tb

def cleanUpTextArea(image):
    try:
        height, width = image.shape[:2]
        res = cv2.resize(image, (3 * width, 3 * height), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        denoise = cv2.fastNlMeansDenoising(blur)
        thresh = cv2.threshold(denoise, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        final = cv2.GaussianBlur(thresh, (5, 5), 0)
        return final
    except Exception as e:
        print("unable to display text block")
        return e


def correctSkew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    print("[INFO] angle: {:.3f}".format(angle))
    return rotated

def RunSpellCheck(InputString):
    """
    Takes a string and attempts to correct spelling on each word
    :param InputString:
    :return: suggested auto corrected spelling
    """
    words = InputString.split(' ')
    OutPutString = ''
    for word in words:
        OutPutString += spell(word) + ' '
    return OutPutString

def RemoveNonUtf8BadChars(line):
    """Remove junk characters from OCR text output.
    Tesseract is pretty good, but sometimes it spits out a bunch of garbage characters
    """
    return "".join([ch for ch in line if ch in printable])

def ocrImage(image, extractBadChars=False, spellCheck=False):
    text = pytesseract.image_to_string(image)
    if extractBadChars:
        text = RemoveNonUtf8BadChars(text)
    if spellCheck:
        text = RunSpellCheck(text)
    return text
