import cv2
import testMxnet
import matplotlib
matplotlib.use('agg')


vidcap = cv2.VideoCapture(0)
count = 0
input_width = 300
input_height = 300
data_shape = 300
success, image = vidcap.read()
yscale = float(image.shape[0] / input_height)
xscale = float(image.shape[1] / input_width)


while success:
    success, image = vidcap.read()

    topn = testMxnet.predict(image, (input_width, input_height))
    xmax, xmin  , ymax , ymin= [int(x * data_shape) for x in topn[0][1:5]]

    xmin = int(xscale * xmin) + int((xmin - input_width / 2) + input_width / 2)
    ymin = int(yscale * ymin)
    xmax = int(xscale * xmax) + int((xmax - input_width / 2) + input_width / 2)
    ymax = int(yscale * ymax)
    #print(topn[0][1:5])
    print("%s %s %s %s " % (xmin, xmax, ymin, ymax))
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 165, 20), 4)
    cv2.imshow('frame', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vidcap.release()
vidcap.release()
cv2.destroyAllWindows()
