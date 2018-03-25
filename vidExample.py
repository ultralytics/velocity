from functions.fcns import *  # local functions
from functions.plots import *  # local plots
import scipy.io
import time
import imutils
import numpy as np
import cv2  # pip install opencv-python, pip install opencv-contrib-python

isVideo = True
pathname = '/Users/glennjocher/Downloads/DATA/VSM/2018.3.11/'
if isVideo:
    filename = pathname + 'IMG_4134.MOV'
    startFrame = 19  # 0 indexed
    readSpeed = 1  # read every # frames. ref = 1 reads every frame, ref = 2 skips every other frame, etc.
    n = 10  # number of frames to read
    frames = np.arange(0, n, 1) * readSpeed + startFrame  # video frames to read
    # filename = '/Users/glennjocher/Downloads/DATA/VSM/2018.3.11/IMG_411%01d.JPG'
    cam, cap = getCameraParams(filename, platform='iPhone 6s')
    print('Starting image processing on ' + filename + ' ...')
    q = np.array([[3761.4, 1503],
                  [3816.3, 1634.4],
                  [3513.3, 1699.6],
                  [3465.7, 1559.1]]).astype('float32')
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
    cam, cap_unused = getCameraParams(filename, platform='iPhone 6s')
    q = np.array([[3761.4, 1503],
                  [3816.3, 1634.4],
                  [3513.3, 1699.6],
                  [3465.7, 1559.1]]).astype('float32')

# Define camera and car information matrices
K = cam['IntrinsicMatrix']
A = np.zeros([n, 14])  # [xyz, rpy, xyz_ecef, lla, t, number](nx14) camera information
B = np.zeros([n, 14])  # [xyz, rpy, xyz_ecef, lla, t, number](nx14) car information
P = np.empty([5, 204, n])  # KLT [x y valid]
P[:] = np.nan
vg = np.zeros_like(P[2, :, 0]) == 1

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
    scale = 1
    im = imutils.resize(im, width=(3840 * scale))

    # KLT tracking
    if i == 0:
        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=200, qualityLevel=0.1, minDistance=2, blockSize=15)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=11,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        q *= scale
        bbox = cv2.boundingRect(q)  # [x0 y0 width height]
        roi = im[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
        p = cv2.goodFeaturesToTrack(roi, mask=None, useHarrisDetector=True, **feature_params).squeeze()

        # #cvb = cv2.BRISK.create()#thresh=40, octaves=4)
        # #bp = cvb.detect(roi)
        # surf = cv2.xfeatures2d.SURF_create(400)
        # bp = surf.detect(roi, None)
        # nbp=len(bp)
        # p = np.zeros([nbp,2],dtype='float32')
        # for j in np.arange(nbp):
        #     p[j,:] = bp[j].pt

        p[:, 0] += bbox[0]
        p[:, 1] += bbox[1]

        mat = scipy.io.loadmat('/Users/glennjocher/Documents/PyCharmProjects/Velocity/data/pc7.mat')
        p = mat['pc']

        p = np.concatenate((q, p), axis=0)
        vi = np.ones(p.shape[0])  # valid points in image i
        vg[0:vi.size] = True  # P[3,] valid points globally

        t, R, residuals, p_ = estimatePlatePosition(K, p[0:4, :], worldPointsLicensePlate(), im)
        p_w = image2world(K, R, t, p)

        # initialize
        fbe = 0  # forward-backward error
        dt = 0
        imfirst = im
        dr = 0
        r = 0
        speed = 0
        t, R, residuals, p_ = estimatePlatePosition(K, p, p_w, im)
    else:
        # update
        p, vi, fbe = KLTwarp(im, im0, p, p0, **lk_params)

        # Get plate position
        t, R, residuals, p_ = estimatePlatePosition(K, p, p_w, im)

        p = p[vi.ravel() == 1]
        p_ = p_[vi.ravel() == 1]
        p_w = p_w[vi.ravel() == 1]
        vg[vg] = vi.ravel()
        dt = A[i, 12] - A[i - 1, 12]
        dr = np.linalg.norm(t - B[i - 1, 0:3])
        r += dr
        speed = dr / dt * 3.6  # m/s to km/h

    # Print image[i] results
    proc_dt[i] = time.time() - tic
    print('%13g%13.3f%13g%13.3f%13.3f%13.3f%13.2f%13.2f%13.1f' %
          (frames[i], proc_dt[i], p.shape[0], np.mean(residuals), dt, A[i, 12], dr, r, speed))

    B[i, 0:3] = t  # car xyz
    P[0, vg, i] = p[:, 0]  # x
    P[1, vg, i] = p[:, 1]  # y
    P[2, :, i] = vg  # status
    P[3, vg, i] = p_[:, 0]  # x
    P[4, vg, i] = p_[:, 1]  # y
    im0 = im
    p0 = p

if isVideo:
    # cv2.destroyAllWindows()  # Closes all cv2 frames
    cap.release()  # Release the video capture object

# Plot All
plotresults(cam, im // 2 + imfirst // 2, P, bbox=bbox)  # // is integer division

dta = time.time() - proc_tstart
print('\nProcessed ', n, ' images: ', frames[:], '\nElapsed Time: ', round(dta, 3), 's (', round(n / dta, 2), ' FPS), ',
      round(dta / n, 3), 's per image\n\n', sep='')
