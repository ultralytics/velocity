from fcns import *
import time
import scipy.io


# import plotly.offline as py  # py.tools.set_credentials_file(username='glenn.jocher', api_key='Hcd6P8v69EAWUrdZDmpU')
# import plotly.graph_objs as go

# pip install --upgrade numpy scipy opencv-python exifread bokeh tensorflow

# @profile
def vidExamplefcn():
    n = 158  # number of frames to read
    isVideo = True
    patha = '/Users/glennjocher/Downloads/DATA/VSM/'
    pathb = '/Users/glennjocher/Google Drive/MATLAB/SPEEDTRAP/'
    if isVideo:
        filename, startframe = patha + '2018.3.11/IMG_4119.MOV', 41  # 20km/h
        # filename, startframe = patha + '2018.3.11/IMG_4134.MOV', 19  # 40km/h
        # filename, startframe = patha + '2018.3.30/IMG_4238.m4v', 8  # 60km/h
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
    B = np.zeros([n, 14], dtype=np.float32)  # [xyz, rpy, xyz_ecef, lla, t, number](nx14) car information
    S = np.zeros([n, 9], dtype=np.float32)  # stats
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
            B[i, 12] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # access CAP_PROP *before* reading!
            B[i, 13] = cap.get(cv2.CAP_PROP_POS_FRAMES)
            success, imbgr = cap.read()  # read frame
            im = cv2.cvtColor(imbgr, cv2.COLOR_BGR2GRAY)
        else:
            success, im = True, cv2.imread(imagename[i], 0)  # 0 second argument = grayscale
            exif = importEXIF(imagename[i])
            B[i, 9:13] = fcnEXIF2LLAT(exif)
        if not success:
            break

        # Scaling and histogram equalization
        scale = 1
        if scale != 1:
            im = cv2.resize(im, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # KLT tracking
        if i == 0:
            q *= scale
            boxa = boundingRect(q, im.shape, border=(0, 0))
            boxb = boundingRect(q, im.shape, border=(1200, 800))
            roi = im[boxb[2]:boxb[3], boxb[0]:boxb[1]]
            p = cv2.goodFeaturesToTrack(roi, 1000, 0.01, 0, blockSize=5, useHarrisDetector=True).squeeze() + np.float32(
                [boxb[0], boxb[2]])
            p = cv2.cornerSubPix(im, p, (5, 5), (-1, -1),
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001))
            p = np.concatenate((q, p))
            t, R, residuals, _ = estimateWorldCameraPose(K, q, worldPointsLicensePlate(), findR=True)
            p3 = addcol0(image2world(K, R, t, p).astype(float)) @ R + t
            R = np.eye(3)
            # residual = p - world2image(K, np.eye(3), np.array([0, 0, 0]), p3 + t)

            # initialize
            vg = np.empty(p.shape[0], dtype=bool)  # valid global points
            vg[:] = True
            vp = insidebbox(p, boxa)
            p_ = p[vp]
            P = np.empty([5, p.shape[0], n], dtype=np.float32)  # KLT [x y valid]
            P[:] = np.nan

            imfirst, dt, dr, r, speed, t0, B[0, 0:3] = im, 0, 0, 0, 0, B[0, 12], t
            im0_small = None
        else:
            # update
            p, v, im0_small = KLTmain(im, im0, im0_small, p)
            vg[vg] = v
            vp = vp & vg

            # Get plate position
            t, R, residuals, p_ = estimateWorldCameraPose(K, p[vp[vg]], p3[vp], R=R, findR=False)

            dt = B[i, 12] - B[i - 1, 12]
            dr = norm(t + B[0, 0:3] - B[i - 1, 0:3])
            r += dr
            speed = dr / dt * 3.6  # m/s to km/h

            # if i > 1:
            # speed = (norm(t - B[i - 2, 0:3])) / (B[i, 12] - B[i - 2, 12]) * 3.6  # mean over past 2 images

            B[i, 3:6] = t
            B[i, 0:3] = B[0, 0:3] + t
            del im0

        # plots
        # py.plot([go.Histogram(x=residuals, nbinsx=30)], filename='basic histogram')

        P[0, vg, i] = p[:, 0]  # x
        P[1, vg, i] = p[:, 1]  # y
        P[2, vp, i] = p_[:, 0]  # x_proj
        P[3, vp, i] = p_[:, 1]  # y_proj
        P[4, vg, i] = i
        im0 = im

        if True and i == 5:
            tmsv, p3hat = fcnMSV1_t(K, P, B, vg, i)
            p3[vg] = p3hat - t
            vp = vg  # enable all points now

        # Print image[i] results
        proc_dt[i] = time.time() - tic
        S[i, :] = (i, proc_dt[i], vg.sum(), residuals.mean(), dt, B[i, 12] - t0, dr, r, speed)
        print('%13g%13.3f%13g%13.3f%13.3f%13.3f%13.2f%13.2f%13.1f' % tuple(S[i, :]))

        # imrgb = cv2.cvtColor(imbgr,cv2.COLOR_BGR2RGB)
        # plots.imshow(cv2.cvtColor(imrgb,cv2.COLOR_BGR2HSV_FULL)[:,:,0])
    if isVideo:
        cap.release()  # Release the video capture object

    dta = time.time() - proc_tstart
    print('\nSpeed = %.2f +/- %.2f km/h' % (S[1:, 8].mean(), S[1:, 8].std()))
    print('Residuals = %.3f +/- %.3f pixels' % (S[1:, 3].mean(), S[1:, 3].std()))
    print('Processed %g images: %s in %.2fs (%.2ffps)\n' % (n, frames[:], dta, n / dta))

    plots.plotresults(cam, im // 2 + imfirst // 2, P, S, B, bbox=boxb)  # // is integer division


vidExamplefcn()
