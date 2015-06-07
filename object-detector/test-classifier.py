from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('-i', "--image", help="Path to the test image", required=True)
args = vars(parser.parse_args())

im = imread(args["image"], as_grey=False)
min_wdw_sz = (100, 40)
step_size = (10, 10)
downscale = 1.25

clf = joblib.load("../data/svm.model")

def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0) 
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

detections = []
scale = 0

for im_scaled in pyramid_gaussian(im, downscale=downscale):
    if im_scaled.shape[0] < min_wdw_sz[0] or im_scaled.shape[1] < min_wdw_sz[1]:
        break
    for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
        if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
            continue
        fd = hog(im_window, orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(3, 3), visualise=False, normalise=True)
        pred = clf.predict(fd)
        if pred == 1:
            print "Detection at {} {}".format(x, y)
            detections.append((x, y, scale))
        clone = im_scaled.copy()
        cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y + im_window.shape[0]), (255, 255, 255))
        cv2.imshow("Image", clone)
        cv2.waitKey(30)
    scale+=1

print detections
for (x, y, scale) in detections:
    cv2.rectangle(im, (x, y), (x + int(min_wdw_sz[0]*(downscale**scale)), y +
        int(min_wdw_sz[1]*(downscale**scale))), (255, 255, 255))
cv2.imshow("Final Detections", im)
cv2.waitKey()
