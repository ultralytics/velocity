from utils.common import *
from utils.strings import *
from utils.transforms import *
from utils.images import *
from utils.KLT import *
from utils.MSV import *
from utils.NLS import *
import plots


# @profile
def vidExamplefcn():
    isVideo = False
    patha = './data/'
    pathb = './matlab/'
    if isVideo:
        # filename, startframe = patha + 'IMG_4119.MOV', 41  # 20km/h 2018.3.11
        filename, startframe = patha + 'IMG_4134.MOV', 19  # 40km/h 2018.3.11
        # filename, startframe = patha + 'IMG_4238.MOV', 8  # 60km/h 2018.3.30
        readSpeed = 1  # read every # frames
        n = 20  # number of frames to read
        frames = np.arange(n) * readSpeed + startframe  # video frames to read
    else:
        frames = np.arange(4122, 4134)
        n = len(frames)
        imagename = []
        for i in frames:
            imagename.append(patha + 'IMG_' + str(i) + '.JPG')
        filename = imagename[0]

    cam, cap = getCameraParams(filename, platform='iPhone 6s')
    mat = scipy.io.loadmat(pathb + cam['filename'] + '.mat')
    q = mat['q'].astype(np.float32)

    # Define camera and car information matrices
    cput0 = time.time()
    K = cam['IntrinsicMatrix']
    B = np.zeros([n, 14], dtype=np.float32)  # [xyz, rpy, xyz_ecef, lla, t, number](nx14) car information
    S = np.zeros([n, 9], dtype=np.float32)  # stats
    _ = np.linalg.inv(np.random.rand(3, 3) @ np.random.rand(3, 3))  # for profiling purposes

    # Iterate over images
    proc_dt = np.zeros([n, 1])
    print('Starting image processing on %s ...' % filename)
    print(('\n' + '%13s' * 9) * 2 % ('image', 'procTime', 'pointTracks', 'metric', 'dt', 'time', 'dx', 'distance',
                                     'speed', '#', '(s)', '#', '(pixels)', '(s)', '(s)', '(m)', '(m)', '(km/h)'))
    for i in range(n):
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
                    for j in range(df - 1):
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
            boxb = boundingRect(q, im.shape, border=(700, 500))
            roi = im[boxb[2]:boxb[3], boxb[0]:boxb[1]]
            p = cv2.goodFeaturesToTrack(roi, 1000, 0.01, 0, blockSize=5, useHarrisDetector=True).squeeze() + \
                np.float32([boxb[0], boxb[2]])
            p = cv2.cornerSubPix(im, p, (5, 5), (-1, -1),
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001))
            p = np.concatenate((q, p))

            t, R, residuals, _ = estimateWorldCameraPose(K, q, worldPointsLicensePlate(), findR=True)
            p3 = addcol0(image2world(K, R, t, p).astype(float)) @ R + t
            R = np.eye(3)
            B[0, 0:3] = t
            # residual = p - world2image(K, np.eye(3), np.array([0, 0, 0]), p3 + t)

            # initialize
            vg = np.ones(p.shape[0], dtype=bool)  # valid global points
            vp = insidebbox(p, boxa)
            p_ = p[vp]
            P = np.empty([5, p.shape[0], n], dtype=np.float32)  # KLT [x y valid]
            P[:] = np.nan

            imfirst, im0_small, dt, dr, r, t0 = im, None, np.nan, 0, 0, B[0, 12]
        else:
            # KLT
            p, v, im0_small = KLTmain(im, im0, im0_small, p)
            vg[vg] = v
            vp = vp & vg

            # fit plate
            t, R, residuals, p_ = estimateWorldCameraPose(K, p[vp[vg]], p3[vp], R=R, findR=False)

            # save results
            dt = B[i, 12] - B[i - 1, 12]
            dr = norm(t + B[0, 0:3] - B[i - 1, 0:3])
            r += dr
            B[i, 3:6] = t
            B[i, 0:3] = B[0, 0:3] + t
            del im0

        # py.plot([go.Histogram(x=residuals, nbinsx=30)])

        P[0:2, vg, i] = p.T  # xy
        P[2:4, vp, i] = p_.T  # xy_proj
        P[4, vg, i] = i
        im0 = im

        msvFrame = 5
        if True and i == msvFrame:
            # B[0:i, 3:6], p3[vg] = fcnNLS_batch(K, P[:,:,0:i], p3, B[0:i, 3:6])
            tmsv, p3hatmsv = fcnMSV1_t(K, P, B, vg, i)
            p3[vg] = p3hatmsv - t
            vp = vg  # enable all points now

        # Print image[i] results
        proc_dt[i] = time.time() - tic
        S[i, :] = (i, proc_dt[i], vg.sum(), residuals, dt, B[i, 12] - t0, dr, r, dr / dt * 3.6)
        print('%13g%13.3f%13g%13.3f%13.3f%13.3f%13.2f%13.2f%13.1f' % tuple(S[i, :]))

        # imrgb = cv2.cvtColor(imbgr,cv2.COLOR_BGR2RGB)
        # plots.imshow(cv2.cvtColor(imrgb,cv2.COLOR_BGR2HSV_FULL)[:,:,0])
        im_gaussian = cv2.GaussianBlur(im, (3, 3), 0)
        im_canny = cv2.Canny(im_gaussian, 100, 200)
        # plots.imshow(cv2.GaussianBlur(im_canny, (9, 9), 0))

    if isVideo:
        cap.release()  # Release the video capture object

    dta = time.time() - cput0
    print('\nSpeed = %.2f +/- %.2f km/h\nRes = %.3f pixels' % (S[1:, 8].mean(), S[1:, 8].std(), S[1:, 3].mean()))
    print('Processed %g images: %s in %.2fs (%.2ffps)\n' % (n, frames[:], dta, n / dta))

    # Post-track process
    if False:
        r = 0
        for i in range(n):
            # j = max(i, msvFrame)
            vg = ~np.isnan(P[3, :, i])
            p = P[0:2, vg, i].T
            t, R, residuals, p_ = estimateWorldCameraPose(K, p, p3[vg], R=R, findR=False)
            if i > 0:
                dt = B[i, 12] - B[i - 1, 12]
                dr = norm(t + B[0, 0:3] - B[i - 1, 0:3])
            else:
                dt = np.nan
                dr = 0
            r += dr
            B[i, 3:6] = t
            B[i, 0:3] = B[0, 0:3] + t
            P[0:2, vg, i] = p.T  # xy
            P[2:4, vg, i] = p_.T  # xy_proj
            P[4, vg, i] = i
            S[i, :] = (i, proc_dt[i], vg.sum(), residuals, dt, B[i, 12] - t0, dr, r, dr / dt * 3.6)
            print('%13g%13.3f%13g%13.3f%13.3f%13.3f%13.2f%13.2f%13.1f' % tuple(S[i, :]))
        print('\nSpeed = %.2f +/- %.2f km/h\nRes = %.3f pixels' % (S[1:, 8].mean(), S[1:, 8].std(), S[1:, 3].mean()))
        print('Processed %g images: %s in %.2fs (%.2ffps)\n' % (n, frames[:], dta, n / dta))

    # Batch NLS
    # Batch2 may work better with tripod, currently camera is tilting a little
    if False:
        B[:, 3:6], _ = fcnNLS_batch(K, P, p3, B[:, 3:6])
        S[1:, 6] = norm(B[1:, 3:6] - B[:-1, 3:6], axis=1)
        S[:, 7] = np.cumsum(S[:, 6])
        S[1:, 8] = S[1:, 6] / S[1:, 4] * 3.6
        for i in range(0, n):
            print('%13g%13.3f%13g%13.3f%13.3f%13.3f%13.2f%13.2f%13.1f' % tuple(S[i, :]))
        print('\nSpeed = %.2f +/- %.2f km/h\nResiduals = %.3f pixels' % (
            S[1:, 8].mean(), S[1:, 8].std(), S[1:, 3].mean()))
        print('Processed %g images: %s in %.2fs (%.2ffps)\n' % (n, frames[:], dta, n / dta))

    plots.plotresults(cam, im // 2 + imfirst // 2, P, S, B, bbox=boxb)  # // is integer division


if __name__ == '__main__':
    vidExamplefcn()
