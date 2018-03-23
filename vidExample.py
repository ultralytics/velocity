from functions.fcns import *  # local functions
from functions.plots import *  # local plots
import time
import imutils
import numpy as np
import cv2  # pip install opencv-python

isVideo = True
pathname = '/Users/glennjocher/Downloads/DATA/VSM/2018.3.11/'
if isVideo:
    filename = pathname + 'IMG_4134.MOV'
    startFrame = 19  # 0 indexed
    readSpeed = 2  # read every # frames. ref = 1 reads every frame, ref = 2 skips every other frame, etc.
    n = 2  # number of frames to read
    frames = np.arange(0, n, 1) * readSpeed + startFrame  # video frames to read
    # filename = '/Users/glennjocher/Downloads/DATA/VSM/2018.3.11/IMG_411%01d.JPG'
    cap = cv2.VideoCapture(filename)
    cam = getCameraParams(filename, platform='iPhone 6s')
    print('Starting image processing on ' + filename + ' ...')
else:
    # frames = np.arange(4122,4133+1)
    # imagename = '/Users/glennjocher/Downloads/DATA/VSM/2018.3.11/IMG_4124.JPG'
    frames = np.array([4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133])
    n = 10  # frames.size
    imagename = []
    for i in frames:
        imagename.append(pathname + 'IMG_' + str(i) + '.JPG')
    filename = imagename[0]
    print('Starting image processing on ' + imagename[0] + ' through ' + imagename[-1] + ' ...')
    cam = getCameraParams(filename, platform='iPhone 6s')

# Define camera and car information matrices
A = np.zeros([n, 14])  # [xyz, rpy, xyz_ecef, lla, t, number](nx14) camera information
B = np.zeros([n, 14])  # [xyz, rpy, xyz_ecef, lla, t, number](nx14) car information

# Iterate over images
proc_dt = np.zeros([n, 1])
proc_tstart = time.time()
print(('\n' + '%13s' * 9) * 2 % ('image', 'procTime', 'pointTracks', 'metric', 'dt', 'time', 'dx', 'range', 'speed',
                                 '#', '(s)', '#', '(pixels)', '(s)', '(s)', '(m)', '(m)', '(km/h)'))
for i in range(0, n):
    tic = time.time()

    # read image
    if isVideo:
        if i == 0:
            if startFrame != 0:
                cap.set(1, startFrame)
        else:
            df = frames[i] - frames[i - 1]
            if df > 1:
                # cap.set(1, frames[i])
                for j in range(0, df - 1):
                    cap.read()  # skip frames
        A[i, 12] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # access CAP_PROP *before* reading!
        A[i, 13] = cap.get(cv2.CAP_PROP_POS_FRAMES)
        success, im = cap.read()  # read frame
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        success = True
        im = cv2.imread(imagename[i], 0)  # 0 second argument = grayscale
        exif = importEXIF(imagename[i])
        A[i, 9:13] = fcnEXIF2LLAT(exif)
        if i == 0:
            t0 = A[0, 12]
        A[i, 12] -= t0
    if not success:
        break
    im = imutils.resize(im, width=(960 * 2))

    # KLT tracking
    if i == 0:
        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=100, qualityLevel=0.2, minDistance=10, blockSize=17)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=10,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        mask = np.zeros_like(im)
        mask[720:880, 1700:2000] = 1
        p0 = cv2.goodFeaturesToTrack(im, mask=mask, **feature_params)

        # initialize
        err = 0
        dt = 0
    else:
        p0, st, err = cv2.calcOpticalFlowPyrLK(imm1, im, pm1, None, **lk_params)
        p0 = p0[st.ravel() == 1]

        dt = A[i, 12] - A[i - 1, 12]

    # Print image[i] results
    proc_dt[i] = time.time() - tic
    print('%13g%13.3f%13g%13.1f%13.3f%13.3f%13.2f%13.2f%13.1f' %
          (frames[i], proc_dt[i], p0.shape[0], np.mean(err), dt, A[i, 12], 0, 0, 0))

    # Plot 1
    if i == 1:
        plot1image(cam, im // 2 + imm1 // 2, p0)  # // is integer division

    pm1 = p0.copy()
    imm1 = im.copy()
if isVideo:
    cv2.destroyAllWindows()  # Closes all the frames
    cap.release()  # Release the video capture object

# Plot All

dta = time.time() - proc_tstart
print('\nProcessed ', n, ' images: ', frames[:], '\nElapsed Time: ', round(dta, 3), 's (', round(n / dta, 2), ' FPS), ',
      round(dta / n, 3), 's per image\n\n', sep='')
