import cv2
import faces as fc
import numpy as np

from sklearn.externals import joblib

clf = joblib.load('gavin.pkl')
pca = joblib.load('pca.pkl')

def detect_face(f, c):
    """Detect faces."""

    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

    detected = c.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    append_image = f

    for (x, y, w, h) in detected:
        cv2.rectangle(append_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (x, y, w, h) in detected:
        image = gray[y:(y+h), x:(x+w)]
        r = float(5) /image.shape[1]
        dim = (5, int(image.shape[0] * r))
        resize = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        flat = resize.flatten().reshape(1, -1)
        try:
            X_test_pca = pca.transform(flat)
            if clf.predict(X_test_pca) == 1:
                cv2.putText(
                    append_image,
                    'Gavin!',
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,0,0),
                    2
                )
        except:
            pass

    return append_image


def extract_face(f, c):
    """Extract faces."""

    extracted = list()

    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

    detected = c.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    append_image = f

    for (x, y, w, h) in detected:
        extracted.append(f[y:(y+h), x:(x+w)])

    return extracted


def main():
    cascPath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)

    video_capture = cv2.VideoCapture(0)

    i = 0

    while True:
        ret, frame = video_capture.read()

        detected_frame = detect_face(frame, faceCascade)

        cv2.imshow('Video', detected_frame)

        listener = cv2.waitKey(1)

        if listener & 0xFF == ord('q'):
            break
        if listener & 0xFF == ord('t'):
            faces = extract_face(frame, faceCascade)
            for face in faces:
                cv2.imwrite('glass-' + str(i) + '.jpg', face)
                i += 1

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
