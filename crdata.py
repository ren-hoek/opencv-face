import os
import faces as fc
import numpy as np


def extract_string(s, a, t=True):
    """Extract end of string.

    Inputs:
        s: string
        a: character to find last/first occurence
        t: True extract end, False extract start
    Output:
        o: string before/after last/first occurence
    """

    if t:
        o = s[s.rfind(a)+1:]
    else:
        o = s[:s.find(a)]

    return o

def create_arrays(i):
    """Create numpy arrays for modelling.

    Convert list of tuples (label, features) into a tuple of
    numpy arrays (np.label, np.featues)

    Inputs:
        i: list of tuples (label, features)
    Outputs:
        o: tuple of numpy arrays
    """
    no_images = len(i)
    no_features = len(i[0][1][0])


    image_array = np.empty(shape = [no_images, no_features])
    image_res = np.empty(shape = [no_images])

    for j, array  in enumerate(i):
        image_res[j] = array[0]
        for k, value in enumerate(array[1][0]):
            image_array[j][k] = value

    return (image_res, image_array)

def create_data(d, n, b=''):
    """Create image dataset.

    Inputs:
        d: root directory of images
        n: output pixel size
        b: folder name for binary classifier (default non-binary)
    Output:
        o: tuple of (np.label_index, np.featureset)
    """

    f = list()

    label_index = 0
    for root, dirs, filenames in os.walk(d):
        cls = extract_string(root, "/")
        if b and b != cls:
            label_index = 0
        print cls, label_index
        for fl in filenames:
            image_path = os.path.join(root, fl)
            face = fc.extract_face(image_path)
            resized_face = fc.resize_images(face, n)
            for r in resized_face:
                f.append(
                    (label_index, r.flatten().reshape(1, -1))
                )
        label_index += 1

    o = create_arrays(f)

    return o

def create_labels(d, b=''):
    """Create labels dictionary.

    Inputs:
        d: root directory of images
        b: folder name for binary classifier (default non-binary)
    Output:
        f: dictionary of labels
    """

    f = dict()

    label_index = 0
    if b == '':
        for root, dirs, filenames in os.walk(d):
            cls = extract_string(root, "/")
            f[label_index] = cls
            label_index += 1
    else:
        f[0]=''
        f[1]=b

    return f


def main():
    n = 5

    image_dir = '/home/gavin/Projects/Faces/lfw-test'

    image_data = create_data(image_dir, n, 'gavin')

    for i in image_data:
        print i[0], i[1]

if __name__ == "__main__":
    main()
