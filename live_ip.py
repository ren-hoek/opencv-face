import cv2
import urllib
import numpy as np
import faces as fc
import crmodel as md

def set_commands(i, f, r):
    """Define keyboard commands.

    Define the key commands while the webcam is running
    Inouts:
        i: the loop counter
        f: the video frame
        r: the event listener
    output:
        current status of the cam
    """
    if r & 0xFF == ord('q'):
        i = -1
    if r & 0xFF == ord('t'):
        faces = fc.extract_face(f, True)
        for face in faces:
            cv2.imwrite('glass-' + str(i) + '.jpg', f)
            i += 1
    return i


def run_classifier():
    """Run the webcam."""
    clf = md.read_classifier('joe-jb.pkl')

    video_capture = cv2.VideoCapture(0)

    #stream=urllib.urlopen('http://192.168.43.1:8080/video')
    stream=urllib.urlopen('http://192.168.42.129:8080/video')

    i = 0

    bytes = ''
    while True:
        bytes+=stream.read(1024)
        a = bytes.find('\xff\xd8')
        b = bytes.find('\xff\xd9')
        if a!=-1 and b!=-1:
            jpg = bytes[a:b+2]
            bytes= bytes[b+2:]
            img = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)
            #detected_frame = fc.identify_face(img, clf, True)
            #cv2.imshow('Video', detected_frame)
            cv2.imshow('Video', img)

        listener = cv2.waitKey(1)

        i = set_commands(i, bytes, listener)
        if i == -1:
            break


    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_classifier()
