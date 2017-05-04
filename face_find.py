import cv2

def resize_image(i, p):
    """Resize image."""
    
    imagePath = i

    cascPath = 'haarcascade_frontalface_default.xml'

    faceCascade = cv2.CascadeClassifier(cascPath)

    image = cv2.imread(imagePath)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print gray.shape

    r = float(p) / gray.shape[1]
    dim = (p, int(image.shape[0] * r))

    print dim

    resized = cv2.resize(gray, dim, interpolation = cv2.INTER_AREA)
    return resized

def main():
    resized = resize_image(
        '/home/gavin/Projects/FaceDetect/glass/glass-0.jpg', 
        150
    )
    
    cv2.imshow("", resized)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
