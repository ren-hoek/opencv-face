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


def add_hat(f, c, m, v=False):
    """Add the wally-mask to hits from the model."""

    image = fc.read_image(f, v)

    extracted = list()
    cascPath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detected = fc.extract_objects(gray, faceCascade)

    append_image = image * 1

    overlay = cv2.imread('wally-mask.png',-1)

    orig_mask = overlay[:,:,3]

    orig_mask_inv = cv2.bitwise_not(orig_mask)

    overlay = overlay[:,:,0:3]
    orig_height, orig_width = overlay.shape[:2]

    yf = 130
    xf = 10

    for (x, y, w, h) in detected:
#        cv2.rectangle(append_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image = gray[y:(y+h), x:(x+w)]
        r = float(c['pixel']) /image.shape[1]
        dim = (c['pixel'], int(image.shape[0] * r))
        resize = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        flat = resize.flatten().reshape(1, -1)
        try:
            X_test_pca = c['pca']['fit'].transform(flat)
            found_face = int(c['classifier'].predict(X_test_pca))
#            prob = c['classifier'].predict_proba(X_test_pca)[0][1]
#            print prob
            if c['labels'][found_face] == m:
                r = float(w) / orig_width
                dim = (int(w * 1.3), int(orig_height * r))

                y -= (yf * r)
                x -= xf

                ov_image = cv2.resize(overlay, dim, interpolation = cv2.INTER_AREA)
                mask = cv2.resize(orig_mask, dim, interpolation = cv2.INTER_AREA)
                mask_inv = cv2.resize(orig_mask_inv, dim, interpolation = cv2.INTER_AREA)

                try:
                    roi = append_image[y:(y + dim[1]), x:(x + dim[0])]
                    roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
                    roi_fg = cv2.bitwise_and(ov_image,ov_image,mask = mask)
                    dst = cv2.add(roi_bg,roi_fg)
                    append_image[y:(y + dim[1]), x:(x + dim[0])] = dst
                except:
                    pass
        except:
            pass

    return append_image


def run_ip_class(url, c, lab):
    """Run classifier via ip network cam."""

    video_capture = cv2.VideoCapture(0)

    stream=urllib.urlopen(url)

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
            detected_frame = add_hat(img, c, lab, True)
            cv2.imshow("Where's Kevin", detected_frame)

        listener = cv2.waitKey(1)

        i = set_commands(i, bytes, listener)
        if i == -1:
            break


    video_capture.release()
    cv2.destroyAllWindows()

    return True


def run_class(c, lab):
    """Run the webcam."""

    video_capture = cv2.VideoCapture(0)

    i = 0

    while True:
        # print i
        ret, frame = video_capture.read()
        # cv2.imshow('Video', frame)
        detected_frame = add_hat(frame, c, lab, True)
        cv2.imshow("Where's Kevin", detected_frame)

        listener = cv2.waitKey(1)

        i = set_commands(i, frame, listener)
        if i == -1:
            break

    video_capture.release()
    cv2.destroyAllWindows()


def run_classifier():
    """Run the webcam."""

    clf = md.read_classifier('models/gav-other.pkl')
    ip_addr = 'http://192.168.43.1:8080/video'
    #run_ip_class(ip_addr, clf, 'gav')
    run_class(clf, 'gav')


if __name__ == "__main__":
    run_classifier()
