import cv2

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
