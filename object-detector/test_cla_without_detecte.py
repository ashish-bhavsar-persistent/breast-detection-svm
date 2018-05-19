# Import the required modules
from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
import argparse as ap
from nms import nms
from glob import glob
from config import *
test_path='/media/dilligencer/A67A6AF37A6ABFA3/DATASET_BREAST/SVM-data/test/pos/'


if __name__ == '__main__':
    file_list = glob(test_path + '*.pgm')
    total_num=len(file_list)
    clf = joblib.load(model_path)
    correct_num = 0
    for file in file_list:
        im=imread(file,as_grey=True)
        fd, hogimage = hog(im, orientations, pixels_per_cell, cells_per_block, visualise=True)
        fd=[fd]
        pred=clf.predict(fd)
        print(pred)
        print(clf.decision_function(fd))
        if pred == 1:
            correct_num+=1
    print(float(correct_num)/total_num)

