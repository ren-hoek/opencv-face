import cv2


def show_image(i, t=""):
    """Show image."""

    cv2.imshow(t, i)
    cv2.waitKey(0)

    return True

def read_image(f, v=False):
    """Read image from picture or video."""
    if not v:
        image = cv2.imread(f)
    else:
        image = f * 1
    return image

def extract_objects(i, c):
    """Extract object from image."""


    extract_objects = c.detectMultiScale(
        i,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    return extract_objects


def detect_face(f, v=False):
    """Detect faces."""

    """
    if not v:
        image = cv2.imread(f)
    else:
        image = f * 1
    """

    image = read_image(f, v)

    cascPath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detected = extract_objects(gray, faceCascade)

    for (x, y, w, h) in detected:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image


def extract_face(f, v=False):
    """Extract faces."""

    """
    if not v:
        image = cv2.imread(f)
    else:
        image = f * 1
    """

    image = read_image(f, v)

    extracted = list()
    cascPath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detected = extract_objects(gray, faceCascade)

    append_image = f

    for (x, y, w, h) in detected:
        extracted.append(gray[y:(y+h), x:(x+w)])

    return extracted


def identify_face(f, c, v=False):
    """Extract faces."""

    """
    if not v:
        image = cv2.imread(f)
    else:
        image = f * 1
    """

    image = read_image(f, v)

    extracted = list()
    cascPath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascPath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detected = extract_objects(gray, faceCascade)

    append_image = image * 1

    for (x, y, w, h) in detected:
        cv2.rectangle(append_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (x, y, w, h) in detected:
        image = gray[y:(y+h), x:(x+w)]
        r = float(c['pixel']) /image.shape[1]
        dim = (c['pixel'], int(image.shape[0] * r))
        resize = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        flat = resize.flatten().reshape(1, -1)
        try:
            X_test_pca = c['pca']['fit'].transform(flat)
            found_face = int(c['classifier'].predict(X_test_pca))
            cv2.putText(
                append_image,
                c['labels'][found_face] + "!!!",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,0,0),
                2
            )
            """
            if c['classifier'].predict(X_test_pca) == 1:
                cv2.putText(
                    append_image,
                    'Gavin!',
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255,0,0),
                    2
                )
            """
        except:
            pass

    return append_image

def resize_images(i, p):
    """Resize images list."""
    resized = list()
    for image in i:
        r = float(p) / image.shape[1]
        dim = (p, int(image.shape[0] * r))
        resized.append(
            cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        )
    return resized


