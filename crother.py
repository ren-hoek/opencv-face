import os
import random
from shutil import copyfile

def create_other(d):
    """Create labels dictionary.

    Inputs:
        d: root directory of images
        b: folder name for binary classifier (default non-binary)
    Output:
        f: dictionary of labels
    """

    i = 0

    for root, dirs, filenames in os.walk(d):
       for f in filenames:
           if random.random() <0.1:
               img_src = root+"/"+f
               img_dst = "images/other/"+f
               copyfile(img_src, img_dst)
               i += 1

    return i

if __name__ == "__main__":
    i = create_other('/home/gavin/Projects/Faces-Bak/lfw-deepfunneled')
    print i
