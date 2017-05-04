import cv2
import faces as fc
import time
import random
import os, os.path

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


def take_photos(p, c):
    """Run the webcam.

    Function to run the webcam
    Inputs:
        p: root string for the images
        c: no of pictures to take
    Output:
        True if function completes
    """

    video_capture = cv2.VideoCapture(0)

    j = 0
    k = 0
    m = 0
    while True:
        ret, frame = video_capture.read()
        orig_frame = fc.read_image(frame, True)
        detected_frame = fc.detect_face(frame, True)
        cv2.imshow('Video', detected_frame)

        listener = cv2.waitKey(1)

        j += 1
        if k != j / 24 and m < c:
            print j, k, p, m, c
            filename = create_filename(p)
            cv2.imwrite(filename, orig_frame)
            m += 1
        if m == c:
            break

    video_capture.release()
    cv2.destroyAllWindows()

    return True


def create_filename(p):
    """Create a random filename for the image.

    Input:
        p: root path for the image
    Output:
        filename with random number addded to root path
    """
    return p + "-" + str(int(random.random() * 10 ** 8)) + '.jpg'


def main():
    person = raw_input('Who am I taking pictures of?\n')
    pictures = int(raw_input('How many pictures?\n'))

    image_dir = 'images/' + person
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    photo_root = image_dir + "/" + person

    countdown = 3
    print "Pictures starting in"

    while countdown > 0:
        print str(countdown) + "..."
        time.sleep(1)
        countdown -= 1

    take_photos(photo_root, pictures)

if  __name__ == "__main__":
    main()

