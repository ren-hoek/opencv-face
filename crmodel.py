import crdata as dt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib


def create_eigenfaces(d, n, p):
    """Use PCA to create n eigenfaces.

    Inputs:
        d: original data
        n: no of pca vectors required
        p: pixel size of images
    Outputs:
        v: tuple (pca, pca basis vectors (eigenfaces))
    """
    v = dict()
    pca = PCA(n_components=n, whiten=True).fit(d)
    eigen = pca.components_.T.reshape(n, p, p)
    v['fit'] = pca
    v['basis'] = eigen
    return v


def build_pca(a, b, n, p):
    """Build PCA and pca'd train/test.

    Inputs:
        a: train data
        b: test data
        n: no of pca vectors required
        p: pixel size of images
    Outputs:
        tuple (pca_dict, train_pca, test_pca)
    """
    face_pca = create_eigenfaces(a, n, p)

    x_train_pca = face_pca['fit'].transform(a)
    x_test_pca = face_pca['fit'].transform(b)

    return (face_pca, x_train_pca, x_test_pca)


def build_svm(x, y, z, k, p):
    """Build SVM.

    Inputs:
        x: train features
        y: train labels
        z: test features
        k: SVM kernal
        p: parameters for grid search
    output:
        tuple (classifer, test labels)
    """
    clf = GridSearchCV(SVC(kernel=k), p, fit_params={})
    clf = clf.fit(x, y)
    y_pred = clf.predict(z)
    return (clf, y_pred)


def create_train_test(d, s):
    """Split data into train/test.

    Inputs:
        d: tuple (np.label, np.features)
        s: test proportion
    Outputs:
        t: 5-tuple of np array (x_train, x_test, y_train, y_test, av_bright)
    """
    av_bright = d[1].mean(axis=1)[:, np.newaxis]

    image_array_bright_norm = d[1] - av_bright

    x_train, x_test, y_train, y_test = train_test_split(
        image_array_bright_norm,
        d[0],
        test_size=s)

    t = (x_train, x_test, y_train, y_test, av_bright)

    return t


def build_model(p, d, c, v):
    """Build SVM face recognition.

    Inputs:
        p: pixel size for face images
        d: root directory containing labelled faces
        c: name of class for binary classifier (empty string for multi class)
        v: no of pca eigenvectors
    Output:
        m: dictionary describing the model
    """
    m = dict()

    data = dt.create_data(d, p, c)
    labels = dt.create_labels(d, c)
    print labels

    x_train, x_test, y_train, y_test, av_bright = create_train_test(data, 0.50)
    face_pca, x_train_pca, x_test_pca = build_pca(x_train, x_test, v, p)

    param_grid = {
        'C': [1, 5, 10, 50, 100],
        'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
    }


    clf, y_pred = build_svm(x_train_pca, y_train, x_test_pca, 'rbf', param_grid)


    m['labels'] = labels
    m['bright'] = av_bright
    m['pca'] = face_pca
    m['classifier'] = clf
    m['report'] = classification_report(y_test, y_pred)
    m['matrix'] = confusion_matrix(y_test, y_pred)
    m['pixel'] = p

    return m


def save_classifier(c, f):
    """Save classifier to disk.

    Saves the classifier to a pickled object using joblib from sklearn.

    Inputs:
        c: classifer object
        f: path to save classifier
    """
    joblib.dump(c, f)
    return True


def read_classifier(f):
    """Read classifier to disk.

    Reads the classifier from disk using joblib from sklearn.

    Inputs:
        f: path to classifier
    Outputs:
        classifier object
    """
    return joblib.load(f)


def main():
    recog = build_model(25, '/home/gavin/Projects/Faces/images', 'lee', 50)
    print recog['report']
    print recog['matrix']
    save_classifier(recog, 'models/lee2.pkl')

    recog_2 = read_classifier('models/lee2.pkl')
    print recog_2['report']
    print recog_2['matrix']


if __name__ == "__main__":
    main()

