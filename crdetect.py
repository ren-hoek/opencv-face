import cv2
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


def run_cam():
    """Run the webcam."""

    video_capture = cv2.VideoCapture(0)

    i = 0

    while True:
        ret, frame = video_capture.read()
        detected_frame = fc.detect_face(frame, True)
        cv2.imshow('Video', detected_frame)

        listener = cv2.waitKey(1)

        i = set_commands(i, frame, listener)
        print i
        if i == -1:
            break

    video_capture.release()
    cv2.destroyAllWindows()


def run_classifier():
    """Run the webcam."""
    clf = md.read_classifier('models/joe-2-jb.pkl')

    video_capture = cv2.VideoCapture(0)

    i = 0

    while True:
        ret, frame = video_capture.read()
        detected_frame = fc.identify_face(frame, clf, True)
        cv2.imshow('Video', detected_frame)

        listener = cv2.waitKey(1)

        i = set_commands(i, frame, listener)
        if i == -1:
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_classifier()
