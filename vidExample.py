# from fcns import *
# @profile
def vidExamplefcn():
    import time
    from fcns import np, cv2, getCameraParams, worldPointsLicensePlate, importEXIF, fcnEXIF2LLAT, \
        estimatePlatePosition, image2world, KLTwarp, norm3, cam2ned  # local functions
    import scipy.io  # , plots

    isVideo = True
    pathname = '/Users/glennjocher/Downloads/DATA/VSM/2018.3.11/'
    if isVideo:
        filename = pathname + 'IMG_4134.m4v'
        startFrame = 19  # 0 indexed
        readSpeed = 1  # read every # frames. ref = 1 reads every frame, ref = 2 skips every other frame, etc.
        n = 10  # number of frames to read
        frames = np.arange(n) * readSpeed + startFrame  # video frames to read
        # filename = '/Users/glennjocher/Downloads/DATA/VSM/2018.3.11/IMG_411%01d.JPG'
        cam, cap = getCameraParams(filename, platform='iPhone 6s')
        print('Starting image processing on %s ...' % filename)
        mat = scipy.io.loadmat('/Users/glennjocher/Google Drive/MATLAB/SPEEDTRAP/IMG_4134.MOV.mat')
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
        mat = scipy.io.loadmat('/Users/glennjocher/Google Drive/MATLAB/SPEEDTRAP/IMG_4122.JPG.mat')
    q = mat['q'].astype(np.float32)

    # Define camera and car information matrices
    proc_tstart = time.time()
    K = cam['IntrinsicMatrix']
    A = np.zeros([n, 14])  # [xyz, rpy, xyz_ecef, lla, t, number](nx14) camera information
    B = np.zeros([n, 14])  # [xyz, rpy, xyz_ecef, lla, t, number](nx14) car information
    S = np.empty([n, 9])  # stats

    # Iterate over images
    proc_dt = np.zeros([n, 1])
    print(('\n' + '%13s' * 9) * 2 % ('image', 'procTime', 'pointTracks', 'metric', 'dt', 'time', 'dx', 'distance',
                                     'speed', '#', '(s)', '#', '(pixels)', '(s)', '(s)', '(m)', '(m)', '(km/h)'))
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
            bbox = cv2.boundingRect(q)  # [x0 y0 width height]
            roi = im[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            p = cv2.goodFeaturesToTrack(roi, 300, 0.01, 1, useHarrisDetector=False, ).squeeze() + np.float32(bbox[0:2])

            p = np.concatenate((q, p), axis=0)
            vi = np.empty(p.shape[0], dtype=bool)  # valid points in image i
            vi[:] = True
            vg = vi  # P[3,] valid points globally
            P = np.empty([4, p.shape[0], n])  # KLT [x y valid]
            P[:] = np.nan

            t, R, residuals, p_ = estimatePlatePosition(K, p[0:4, :], worldPointsLicensePlate())
            p_w = image2world(K, R, t, p)

            p_ = p
            # t, R, residuals, p_ = estimatePlatePosition(K, p, p_w)

            # initialize
            imfirst = im
            dt = 0
            dr = 0
            r = 0
            speed = 0
            im0_small = cv2.resize(im, (0, 0), fx=1 / 8, fy=1 / 8, interpolation=cv2.INTER_NEAREST)
        else:
            # update
            p, vi, im0_small = KLTwarp(im, im0, im0_small, p0)

            # Get plate position
            t, R, residuals, p_ = estimatePlatePosition(K, p, p_w)

            vg[vg] = vi
            p = p[vi]
            p_ = p_[vi]
            p_w = p_w[vi]
            dt = A[i, 12] - A[i - 1, 12]
            dr = norm3(cam2ned() @ t - B[i - 1, 0:3])
            r += dr
            speed = dr / dt * 3.6  # m/s to km/h
            del im0

        # Print image[i] results
        proc_dt[i] = time.time() - tic
        mr = residuals.sum() / residuals.size
        S[i, :] = (frames[i], proc_dt[i], p.shape[0], mr, dt, A[i, 12], dr, r, speed)
        print('%13g%13.3f%13g%13.3f%13.3f%13.3f%13.2f%13.2f%13.3f' % tuple(S[i, :]))

        B[i, 0:3] = cam2ned() @ t  # car xyz
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

    import plots
    plots.plotresults(cam, im // 2 + imfirst // 2, P, S, B, bbox=bbox)  # // is integer division


vidExamplefcn()
