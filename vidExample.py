from fcns import *

# @profile
def vidExamplefcn():
    import time
    import scipy.io
    import plots

    n = 21  # number of frames to read
    isVideo = True
    patha = '/Users/glennjocher/Downloads/DATA/VSM/'
    pathb = '/Users/glennjocher/Google Drive/MATLAB/SPEEDTRAP/'
    if isVideo:
        # filename, startframe = patha + '2018.3.11/IMG_4119.MOV', 41
        # filename, startframe = patha + '2018.3.11/IMG_4134.MOV', 19
        filename, startframe = patha + '2018.3.30/IMG_4238.m4v', 8
        readSpeed = 1  # read every # frames
        frames = np.arange(n) * readSpeed + startframe  # video frames to read
    else:
        # imagename = patha + '2018.3.11/IMG_4124.JPG'
        # filename = patha + '2018.3.11/IMG_411%01d.JPG'
        frames = np.array([4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133])
        imagename = []
        for i in frames:
            imagename.append(patha + '2018.3.11/IMG_' + str(i) + '.JPG')
        filename = imagename[0]

    cam, cap = getCameraParams(filename, platform='iPhone 6s')
    mat = scipy.io.loadmat(pathb + cam['filename'] + '.mat')
    q = mat['q'].astype(np.float32)

    # Define camera and car information matrices
    proc_tstart = time.time()
    K = cam['IntrinsicMatrix']
    A = np.zeros([n, 14])  # [xyz, rpy, xyz_ecef, lla, t, number](nx14) camera information
    B = np.zeros([n, 14])  # [xyz, rpy, xyz_ecef, lla, t, number](nx14) car information
    S = np.empty([n, 9])  # stats
    _ = np.linalg.inv(np.random.rand(3, 3) @ np.random.rand(3, 3))  # for profiling purposes

    # Iterate over images
    proc_dt = np.zeros([n, 1])
    print('Starting image processing on %s ...' % filename)
    print(('\n' + '%13s' * 9) * 2 % ('image', 'procTime', 'pointTracks', 'metric', 'dt', 'time', 'dx', 'distance',
                                     'speed', '#', '(s)', '#', '(pixels)', '(s)', '(s)', '(m)', '(m)', '(km/h)'))
    for i in range(0, n):
        tic = time.time()
        # read image
        if isVideo:
            if i == 0:
                if startframe != 0:
                    cap.set(1, startframe)
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

        # Scaling and histogram equalization
        scale = 1
        if scale != 1:
            im = cv2.resize(im, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # im = cv2.equalizeHist(im)
        # if i==0:
        #    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(80, 80))
        # im = clahe.apply(im)

        # KLT tracking
        if i == 0:
            q *= scale
            bbox = boundingRect(q, im.shape, border=0)
            roi = im[bbox[2]:bbox[3], bbox[0]:bbox[1]]
            p = cv2.goodFeaturesToTrack(roi, 1000, 0.01, 0, blockSize=5, useHarrisDetector=True).squeeze() + np.float32(
                [bbox[0], bbox[2]])
            p = cv2.cornerSubPix(im, p, (5, 5), (-1, -1),
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001))
            # p = scipy.io.loadmat('/Users/glennjocher/Documents/PyCharmProjects/Velocity/data/pc_v7.mat')['pc']
            p = np.concatenate((q, p), axis=0)
            t, R, residuals, p_ = estimatePlatePosition(K, p[0:4, :], worldPointsLicensePlate())
            p_w = image2world(K, R, t, p)
            p_ = p

            # initialize
            vi = np.empty(p.shape[0], dtype=bool)  # valid points in image i
            P = np.empty([4, p.shape[0], n])  # KLT [x y valid]
            vi[:] = True
            vg = vi  # P[3,] valid points globally
            P[:] = np.nan
            imfirst, dt, dr, r, speed = im, 0, 0, 0, 0
            im0_small = cv2.resize(im, (0, 0), fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_NEAREST)
        else:
            # update
            p, vi, im0_small = KLTwarp(im, im0, im0_small, p0)
            p_w = p_w[vi]
            p = p[vi]
            vg[vg] = vi

            # Get plate position
            t, R, residuals, p_ = estimatePlatePosition(K, p, p_w, t, R)

            dt = A[i, 12] - A[i - 1, 12]
            dr = norm(t - B[i - 1, 0:3])
            r += dr
            speed = dr / dt * 3.6  # m/s to km/h
            del im0

        # Print image[i] results
        proc_dt[i] = time.time() - tic
        mr = residuals.sum() / residuals.size
        S[i, :] = (i, proc_dt[i], p.shape[0], mr, dt, A[i, 12], dr, r, speed)
        print('%13g%13.3f%13g%13.3f%13.3f%13.3f%13.2f%13.2f%13.1f' % tuple(S[i, :]))

        B[i, 0:3] = t  # car xyz
        P[0, vg, i] = p[:, 0]  # x
        P[1, vg, i] = p[:, 1]  # y
        P[2, vg, i] = p_[:, 0]  # x_proj
        P[3, vg, i] = p_[:, 1]  # y_proj
        im0 = im
        p0 = p
    if isVideo:
        cap.release()  # Release the video capture object

    dta = time.time() - proc_tstart
    print('\nSpeed = %.2f +/- %.2f km/h' % (S[1:, 8].mean(), S[1:, 8].std()))
    print('Residuals = %.3f +/- %.3f pixels' % (S[1:, 3].mean(), S[1:, 3].std()))
    print('Processed %g images: %s in %.2fs (%.2ffps)\n' % (n, frames[:], dta, n / dta))

    plots.plotresults(cam, im // 2 + imfirst // 2, P, S, B, bbox=bbox)  # // is integer division


vidExamplefcn()
