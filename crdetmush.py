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
    i += 1
    if r & 0xFF == ord('q'):
        i = -1
    return i


def add_hat(f, c, v=False):
    """Extract faces."""

    """
    if not v:
        image = cv2.imread(f)
    else:
        image = f * 1
    """

    image = fc.read_image(f, v)

    extracted = list()
    cascPath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detected = fc.extract_objects(gray, faceCascade)

    append_image = image * 1

    # Load our overlay image: mustache.png
    imgMustache = cv2.imread('wally-mask.png',-1)

    # Create the mask for the mustache
    orig_mask = imgMustache[:,:,3]

    # Create the inverted mask for the mustache
    orig_mask_inv = cv2.bitwise_not(orig_mask)

    # Convert mustache image to BGR
    # and save the original image size (used later when re-sizing the image)
    imgMustache = imgMustache[:,:,0:3]
    origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]

    yf = 130
    xf = 0
    for (x, y, w, h) in detected:
        cv2.rectangle(append_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image = gray[y:(y+h), x:(x+w)]
        r = float(c['pixel']) /image.shape[1]
        dim = (c['pixel'], int(image.shape[0] * r))
        resize = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        flat = resize.flatten().reshape(1, -1)
        try:
            X_test_pca = c['pca']['fit'].transform(flat)
            found_face = int(c['classifier'].predict(X_test_pca))
            prob = c['classifier'].predict_proba(X_test_pca)[0][1]
            print prob
            if c['labels'][found_face] == "joe" and prob > 0.0:
                r = float(w) / origMustacheWidth
                dim = (int(w * 1.3), int(origMustacheHeight * r))

                y -= (yf * r)
                x -= xf

                mustache = cv2.resize(imgMustache, dim, interpolation = cv2.INTER_AREA)
                mask = cv2.resize(orig_mask, dim, interpolation = cv2.INTER_AREA)
                mask_inv = cv2.resize(orig_mask_inv, dim, interpolation = cv2.INTER_AREA)

                try:
                    roi = append_image[y:(y + dim[1]), x:(x + dim[0])]
                    roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
                    roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)
                    dst = cv2.add(roi_bg,roi_fg)
                    append_image[y:(y + dim[1]), x:(x + dim[0])] = dst
                except:
                    pass
        except:
            pass

    return append_image

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
    clf = md.read_classifier('models/joe-other.pkl')

    video_capture = cv2.VideoCapture(0)

    i = 0

    while True:
        # print i
        ret, frame = video_capture.read()
        # cv2.imshow('Video', frame)
        detected_frame = add_hat(frame, clf, True)
        cv2.imshow('Video', detected_frame)

        listener = cv2.waitKey(1)

        i = set_commands(i, frame, listener)
        if i == -1:
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_classifier()
