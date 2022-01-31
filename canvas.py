import cv2.cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow import keras

# load cnn model
model = keras.models.load_model('cnn_digits_recognition.h5')
print('Model loaded.')
model.summary()
predicted_label = 0

thickness = 25
canvas_height = 500
canvas_width = 500


def thickness_callback(x):
    global thickness
    thickness = x


# create a canvas
canvas = np.zeros((canvas_height, canvas_width), np.uint8)

# create a named window
cv.namedWindow("canvas")

# create a trackbar to modify thickness
cv.createTrackbar("thickness", "canvas", thickness, 255, thickness_callback)

is_drawing = False


# define a drawing function
def draw(event, x, y, flag, param):
    global is_drawing, thickness, predicted_label
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(canvas, (x, y), thickness, (255, 255, 255), -1)
        is_drawing = True
    elif event == cv.EVENT_MOUSEMOVE and is_drawing == True:
        cv.circle(canvas, (x, y), thickness, (255, 255, 255), -1)
    elif event == cv.EVENT_LBUTTONUP:
        is_drawing = False
        resized_canvas = cv.resize(canvas, (28, 28), interpolation=cv.INTER_CUBIC)
        resized_canvas = cv.blur(resized_canvas, (3, 3))
        cv.imshow("resized_canvas", resized_canvas)

        input_layer = np.expand_dims(resized_canvas, axis=0)

        res = model.predict(input_layer, batch_size=1)
        predicted_label = np.argmax(res[0])
        print("Predicted label: %d" % predicted_label)


# set mouse callback
cv.setMouseCallback("canvas", draw)

# main loop
while True:
    # predict the label
    # down sampling

    # print(input_layer.shape)
    # predict
    # np.expand_dims(input_layer, axis=0)
    # np.expand_dims(input_layer, axis=2)

    # show the canvas
    cv.imshow("canvas", canvas)

    # get input key
    key = cv.waitKey(30)
    # if key == esc, then exit the main loop
    if key == 27:
        break
    elif key == ord('c'):
        canvas = np.zeros((canvas_height, canvas_width), np.uint8)

cv.destroyAllWindows()
