#!/usr/bin/env python3

from threading import Thread, Lock, Semaphore
from queue import Queue
import cv2
import numpy as np
import base64
import time

DEBUG = False
BUF_SIZE = 10               # maximum size of buffer
clipFileName = 'clip.mp4'
frameDelay   = 42           # the answer to everything

# queues
extractBuffer = Queue(BUF_SIZE)     # for frames extracted from .mp4
grayBuffer = Queue(BUF_SIZE)        # for frames converted to grayscale

# queue locks
exLock = Lock()                     # to lock extractBuffer
grayLock = Lock()                   # to lock grayBuffer

# semaphores
extSemaEmpty = Semaphore(BUF_SIZE)    # counting sem for extractBuffer
extSemaFull = Semaphore(0)

graySemaEmpty = Semaphore(BUF_SIZE)  # counting sem for grayBuffer
graySemaFull = Semaphore(0)

# sentinel object to signal job is done
done = object()

# thread safe put into extractBuffer
def extBufPut(image):
    # acquire emtpty extractBuffer cell
    extSemaEmpty.acquire()
    # put frame in buffer
    with exLock:
        extractBuffer.put(image)
    # space taken in buffer
    extSemaFull.release()

# thread safe get from extractBuffer
def extBufGet():
    # get full cell from extract buffer
    extSemaFull.acquire()
    # get next frame from buffer
    with exLock:
        image = extractBuffer.get()
    # space freed in extractBuffer
    extSemaEmpty.release()
    return image

# thread safe put into grayBuffer
def grayBufPut(image):
    # get full cell from grayBuffer buffer
    graySemaEmpty.acquire()
    # put frame in buffer
    with grayLock:
        grayBuffer.put(image)
    # space taken in buffer
    graySemaFull.release()

# thread safe get from grayBuffer
def grayBufGet():
    # get full cell from buffer
    graySemaFull.acquire()
    # get the next frame from the buffer
    with grayLock:
        image = grayBuffer.get()
    # space freed in buffer
    graySemaEmpty.release()
    return image


class ExtractFramesThread(Thread):
    def __init__(self,clipFileName,DEBUG):
        Thread.__init__(self, daemon=False)
        self.clipFileName, self.DEBUG = clipFileName, DEBUG
        self.name = 'extractThread'
        self.start()


    def run(self):
        # Initialize frame count
        count = 0

        # open video file
        vidcap = cv2.VideoCapture(clipFileName)

        # read first image
        success,image = vidcap.read()

        print("Reading frame {} {} ".format(count, success))

        # continue for all frames
        while success:
            # get a jpg encoded frame
            success, jpgImage = cv2.imencode('.jpg', image)

            # encode the frame as base 64 to make debugging easier
            jpgAsText = base64.b64encode(jpgImage)

            # place into extractBuffer safely
            extBufPut(jpgAsText)

            # get next frame
            success,image = vidcap.read()
            count += 1
            print('Reading frame {} {}'.format(count, success))

        # job is done, place sentinel
        extBufPut(done)
        print("Frame extraction complete")
        return


class ConvertToGrayscaleThread(Thread):
    def __init__(self,DEBUG):
        Thread.__init__(self, daemon=False)
        self.name = 'convertThread'
        self.start()


    def run(self):
        # Initialize frame count
        count = 0

        # get first framed from extractBuffer
        frameAsText = extBufGet()

        # continue until sentinel is detected
        while frameAsText is not done:
            print("Converting frame {}".format(count))

            #decode the frame
            jpgRawImage = base64.b64decode(frameAsText)

            # convert the raw frame to a numpy array
            jpgImage = np.asarray(bytearray(jpgRawImage), dtype=np.uint8)

            # decode the jpg
            img = cv2.imdecode(jpgImage, cv2.IMREAD_UNCHANGED)

            # convert the image to grayscale
            grayscaleFrame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # get a jpg encoded frame
            success, jpgImage = cv2.imencode('.jpg', grayscaleFrame)

            #encode the frame as base 64 to make debugging easier
            grayJpgAsText = base64.b64encode(jpgImage)

            # put encoded grayscaleFrame in grayBuffer safely
            grayBufPut(grayJpgAsText)

            # get next frame from extract buffer
            frameAsText = extBufGet()

            count += 1

        # job is done, place sentinel
        grayBufPut(done)
        print("Frame conversion complete")
        return


class DisplayFramesThread(Thread):
    def __init__(self,DEBUG):
        Thread.__init__(self, daemon=False)
        self.name, self.DEBUG = 'displayThread', DEBUG
        time.sleep(0.01)
        self.start()


    def run(self):
        # Initialize frame count and frame
        count = 0

        startTime = time.time()

        # get first grayscale frame
        frameAsText = grayBufGet()

        # continue until sentinel is detected
        while frameAsText is not done:
            # decode the frame
            jpgRawImage = base64.b64decode(frameAsText)

            # convert the raw frame to a numpy array
            jpgImage = np.asarray(bytearray(jpgRawImage), dtype=np.uint8)

            # get a jpg encoded frame
            img = cv2.imdecode( jpgImage ,cv2.IMREAD_UNCHANGED)

            print("Displaying frame {}".format(count))

            # display the image in a window called "video" and wait 42ms
            # before displaying the next frame
            cv2.imshow("Video", img)

            # compute the amount of time that has elapsed
            # while the frame was processed
            elapsedTime = int((time.time() - startTime) * 1000)
            print("Time to process frame {} ms".format(elapsedTime))

            # determine the amount of time to wait, also
            # make sure we don't go into negative time
            timeToWait = max(1, frameDelay - elapsedTime)
            if cv2.waitKey(timeToWait) and 0xFF == ord("q"):
                break

            # get the start time for processing the next frame
            startTime = time.time()

            count += 1

            # get frame from grayBuffer safely
            frameAsText = grayBufGet()

        print("\nFinished displaying all frames\nExiting...\n")
        # cleanup the windows
        cv2.destroyAllWindows()
        return


if __name__ == '__main__':
    ExtractFramesThread(clipFileName,DEBUG)
    ConvertToGrayscaleThread(DEBUG)
    DisplayFramesThread(DEBUG)
