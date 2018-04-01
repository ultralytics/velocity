import math

import cv2  # pip install opencv-python, pip install opencv-contrib-python
import numpy as np

# from scipy import interpolate

# Set options
# pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5


# np.set_printoptions(linewidth=320, formatter={'float_kind': '{:21.15g}'.format})  # format long g, %precision=15
# %precision '%0.8g'

def norm_cols(x):
    n = x.shape[1]
    if n == 3:
        return (x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2) ** 0.5
    elif n == 2:
        return (x[:, 0] ** 2 + x[:, 1] ** 2) ** 0.5
    else:
        return (x ** 2).sum(axis=1) ** 0.5


def norm3(x):
    return math.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)


# Define functions
def worldPointsLicensePlate():
    # Returns x, y coordinates of license plate outline in meters
    # https://en.wikipedia.org/wiki/Vehicle_registration_plate
    # size = np.array([.3725, .1275])  # [0.36 0.13] (m) license plate size (Chile)
    # return np.array([[1, -1],[1, 1],[-1, 1],[-1, -1]]) * (size / 2)
    x = 0.3725 / 2
    y = 0.1275 / 2  # [0.36 0.13] #(m) license plate size (Chile)
    return np.array([[x, -y],
                     [x, y],
                     [-x, y],
                     [-x, -y]], dtype='float32')  # worldPoints


def cam2ned():
    # x_ned(3x5) = R * x_cam(3x5)   - EQUALS -   x_ned(5x3) = x_cam(5x3) * R'
    # +X_ned(NORTH) = +Z_cam(NORTH)
    # +Y_ned(EAST)  = +X_cam(EAST)
    # +Z_ned(DOWN)  = +Y_cam(DOWN)
    return np.array([[0, 0, 1],
                     [1, 0, 0],
                     [0, 1, 0]])  # R


def fcnEXIF2LLAT(E):
    # E = image exif info i.e. E = importEXIF('img.jpg')
    # llat = [lat, long, alt (m), time (s)]
    # MATLAB:  datenum('2018:03:11 15:57:22','yyyy:mm:dd HH:MM:SS') # fractional day since 00/00/000
    # Python:  d = datetime.strptime('2018:03:11 15:57:22', "%Y:%m:%d %H:%M:%S"); datetime.toordinal(d) + 366
    # day = datenum(E['EXIF DateTimeOriginal'] + '.' + E['EXIF SubsecTimeOriginal'], 'yyyy:mm:dd HH:MM:SS.FFF')

    s = E['EXIF DateTimeOriginal']
    s = s.split(' ')[1]
    hour, minute, second = s.split(':')
    day_fraction = float(hour) / 24 + float(minute) / 1440 + float(second) / 86400 + E[
        'EXIF SubSecTimeOriginal'] / 86400000
    # d = datetime.strptime(E['EXIF DateTimeOriginal'], "%Y:%m:%d %H:%M:%S")
    # day = datetime.toordinal(d) + 366
    # day_fraction = d.hour / 24 + d.minute / 1440 + d.second / 86400 + E['EXIF SubSecTimeOriginal'] / 86400000

    llat = np.zeros(4)
    llat[0] = dms2degrees(E['GPS GPSLatitude']) * hemisphere2sign(E['GPS GPSLatitudeRef'])
    llat[1] = dms2degrees(E['GPS GPSLongitude']) * hemisphere2sign(E['GPS GPSLongitudeRef'])
    llat[2] = E['GPS GPSAltitude']
    llat[3] = day_fraction * 86400  # seconds since midnight
    return llat


def dms2degrees(dms):
    # converts GPS [degrees minutes seconds] to decimal degrees
    # dms(1x3)
    return dms[0] + dms[1] / 60 + dms[2] / 3600  # degrees


def hemisphere2sign(x):
    # converts hemisphere strings 'N', 'S', 'E', 'W' to signs 1, -1, 1, -1
    sign = np.zeros_like(x, dtype=float)
    sign[(x == 'N') | (x == 'E')] = 1
    sign[(x == 'S') | (x == 'W')] = -1
    return sign


def filenamesplit(string):
    # splits a full filename string into path, filename, and extension. Example:
    # str = '/Users/glennjocher/Downloads/DATA/VSM/2018.3.11/IMG_4124.JPG'
    # path = '/Users/glennjocher/Downloads/DATA/VSM/2018.3.11/'
    # filename = 'IMG_4124.JPG'
    # extension = '.JPG'
    i = string.rfind('/') + 1
    path = string[0:i]
    filename = string[i:None]
    extension = filename[filename.rfind('.'):None]
    return path, filename, extension


# @profile
def getCameraParams(fullfilename, platform='iPhone 6s'):
    # returns camera parameters and file information structure cam
    # fullfilename: video or image(s) file name(s) i.e. mymovie.mov or IMG_3797.jpg
    # platform: camera name i.e. 'iPhone 6s'
    pathname, filename, extension = filenamesplit(fullfilename)
    isvideo = (extension == '.MOV') | (extension == '.mov') | (extension == '.m4v')

    if platform == 'iPhone 6s':
        # pixelSize = 0.0011905 #(mm) on a side, 12um
        sensorSize_mm = np.array([4.80, 3.60])  # (mm) CMOS sensor
        focalLength_mm = 4.15  # (mm) iPhone 6s from EXIF
        # focalLength_pix= focalLength_mm / sensorSize_mm[0] * width
        # fov = np.degrees([math.atan(width/2/focalLength_pix) math.atan(height/2/focalLength_pix)]*2)  # (deg) FOV
        fov = np.degrees(np.arctan(sensorSize_mm / 2 / focalLength_mm) * 2)  # (deg) camea field of view

        if isvideo:  # 4k VIDEO 3840x2160
            cap = cv2.VideoCapture(fullfilename)
            kltBlockSize = [51, 51]

            # https://docs.opencv.org/3.1.0/d8/dfe/classcv_1_1VideoCapture.html#aeb1644641842e6b104f244f049648f94
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            # ratio of image to video frame diagonal lengths:
            # https://photo.stackexchange.com/questions/86075/does-the-iphones-focal-length-differ-when-taking-video-vs-photos
            diagonalRatio = math.sqrt(4032 ** 2 + 3024 ** 2) / math.sqrt(3840 ** 2 + 2160 ** 2)

            skew = 0
            focalLength_pix = np.array([3486, 3486]) * diagonalRatio

            if width > height:  # 1 = landscape, 6 = vertical
                orientation = 1
            else:
                orientation = 6
        else:  # 12MP IMAGE 4032x3024
            cap = []
            exif = importEXIF(fullfilename)
            kltBlockSize = [21, 21]

            orientation = exif['Image Orientation']
            width = exif['EXIF ExifImageWidth']
            height = exif['EXIF ExifImageLength']
            fps = 0
            frame_count = 0

            skew = 0
            focalLength_pix = [3486, 3486]
            # focalLength_pix = exif['EXIF FocalLength'] / sensorSize(1) * width

    elif platform == 'iPhone x':
        'fill in here'

    radialDistortion = [0, 0, 0]
    principalPoint = np.array([width, height]) / 2 + 0.5
    IntrinsicMatrix = np.array([[focalLength_pix[0], 0, 0],
                                [skew, focalLength_pix[1], 0],
                                [principalPoint[0], principalPoint[1], 1]], np.float32)

    if orientation == 1:  # 1 = landscape, 6 = vertical
        orientation_comment = 'Horizontal'
    elif orientation == 6:
        orientation_comment = 'Vertical'

    # Define camera parameters
    params = ''  # cameraParameters('IntrinsicMatrix', IntrinsicMatrix, 'RadialDistortion', radialDistortion)

    # Pre-define interpolation grid for use with imwarp or interp2
    # x, y = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    # ixy = np.stack((x.transpose().flatten(), y.transpose().flatten()), axis=0).transpose()  # one-liner
    # ixy = np.ones([x.size, 3], dtype=float)
    # ixy[:, 0] = x.transpose().flatten()
    # ixy[:, 1] = y.transpose().flatten()
    ixy = 0

    cam = {
        'fullfilename': fullfilename,
        'pathname': pathname,
        'filename': filename,
        'extension': extension,
        'isvideo': isvideo,
        'width': width,
        'height': height,
        'sensorSize_mm': sensorSize_mm,
        'focalLength_mm': focalLength_mm,
        'focalLength_pix': focalLength_pix,
        'fov': fov,
        'skew': skew,
        'principalPoint': principalPoint,
        'IntrinsicMatrix': IntrinsicMatrix,
        'radialDistortion': radialDistortion,
        'ixy': ixy,
        'kltBlockSize': kltBlockSize,
        'orientation': orientation,
        'orientation_comment': orientation_comment,
        'fps': fps,
        'frame_count': frame_count,
        'params': params
    }
    return cam, cap


def importEXIF(fullfilename):
    import exifread
    exif = exifread.process_file(open(fullfilename, 'rb'), details=False)
    for tag in exif.keys():
        a = exif[tag].values[:]
        if type(a) == str and a.isnumeric():
            a = float(a)
        if type(a) == list:
            n = a.__len__()
            a = np.asarray(a)
            for i in range(0, n):
                if type(a[i]) == exifread.utils.Ratio:
                    a[i] = float(a[i].num / a[i].den)
            if n == 1:
                a = a[0]
        exif[tag] = a
    # printd(exif)
    return exif


def printd(dictionary):  # print dictionary
    for tag in dictionary.keys():
        print('%40s: %s' % (tag, dictionary[tag]))


def keypoints2points(kp):
    # inputs ORB keypoint strucutre and outputs nx2  nparray
    nk = kp.__len__()
    p = np.zeros([nk, 2], dtype='float32')
    for j in np.arange(nk):
        p[j, :] = kp[j].pt
    return p


# @profile
def orb_match(im0, im1, p0):
    scale = 1 / 2
    im0 = cv2.resize(im0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    im1 = cv2.resize(im1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    orb = cv2.AKAZE_create()
    # orb = cv2.xfeatures2d.SURF_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # bbox = cv2.boundingRect(p0_)  # [x0 y0 width height]
    # im0_roi = im0[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
    # mask = np.zeros_like(im0)
    # mask[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = 1
    [x0, y0] = p0.min(0) * scale - 25
    [x1, y1] = p0.max(0) * scale + 25
    x0 = max(x0, 1).__int__()
    y0 = max(y0, 1).__int__()
    x1 = min(x1, im0.shape[1]).__int__()
    y1 = min(y1, im0.shape[0]).__int__()
    # im0_roi = im0[y0:y1, x0:x1]

    mask = np.zeros_like(im0)
    mask[y0:y1, x0:x1] = 1
    kp0, des0 = orb.detectAndCompute(im0, mask=mask)
    kp1, des1 = orb.detectAndCompute(im1, None)
    matches = bf.match(des0, des1)

    n = len(matches)
    m0 = np.zeros((n, 2))
    m1 = np.zeros((n, 2))
    for j in np.arange(n):
        i0, i1 = matches[j].queryIdx, matches[j].trainIdx
        m0[j] = kp0[i0].pt
        m1[j] = kp1[i1].pt
    # m0 += bbox[0:2]
    # plots.imshow(im0 // 2 + im1 // 2, None, m0, m1)
    return cv2.estimateAffine2D(m0 / scale, m1 / scale, method=cv2.RANSAC)  # 2x3, better results


# @profile
def KLTregional(im0, im, p0, T, lk_param, fbt=1.0, translateFlag=False):
    T = T.astype('float32')
    # 1. Warp current image to past image frame
    # im_warped_0 = cv2.warpAffine(im, T23, (int(im.shape[1]/2), int(im.shape[0]/2)),flags=cv2.WARP_INVERSE_MAP)
    [x0, y0] = p0.min(0) - 50
    [x1, y1] = p0.max(0) + 50
    x0 = max(x0, 1).__int__()
    y0 = max(y0, 1).__int__()
    x1 = min(x1, im.shape[1]).__int__()
    y1 = min(y1, im.shape[0]).__int__()
    im0_roi = im0[y0:y1, x0:x1]
    xy0 = np.float32([x0, y0])
    p0_roi = p0 - xy0

    if translateFlag:
        dx = T[2, 0].__int__()
        dy = T[2, 1].__int__()
        im_warped_0 = im[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    else:
        x, y = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        ixy = np.ones([x.size, 3], np.float32)
        ixy[:, 0] = x.ravel()
        ixy[:, 1] = y.ravel()
        ixy_ = ixy @ T
        x_ = ixy_[:, 0].reshape(x.shape)
        y_ = ixy_[:, 1].reshape(x.shape)
        # x__, y__ = cv2.convertMaps(x_,y_,cv2.CV_32FC2)
        im_warped_0 = cv2.remap(im, x_, y_, cv2.INTER_LINEAR)  # current image ROI mapped to previous image

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # im0_roi = clahe.apply(im0_roi)
    # im_warped_0 = clahe.apply(im_warped_0)

    im0_roi = cv2.equalizeHist(im0_roi)
    im_warped_0 = cv2.equalizeHist(im_warped_0)

    # _, pyr1 = cv2.buildOpticalFlowPyramid(im0_roi, winSize=lk_param['winSize'], maxLevel=lk_param['maxLevel'], withDerivatives=True)
    # _, pyr2 = cv2.buildOpticalFlowPyramid(im_warped_0, winSize=lk_param['winSize'], maxLevel=lk_param['maxLevel'], withDerivatives=True)

    # run klt tracker forward-backward
    pa, va, _ = cv2.calcOpticalFlowPyrLK(im0_roi, im_warped_0, p0_roi, None, **lk_param)
    pb, vb, _ = cv2.calcOpticalFlowPyrLK(im_warped_0, im0_roi, pa, None, **lk_param)
    fbe = norm_cols(pb - p0_roi)
    v = (va.ravel() == 1) & (vb.ravel() == 1) & (fbe < fbt)  # forward-backward error threshold

    # convert p back to im coordinates
    if translateFlag:
        p = pa + (xy0 + [dx, dy]).astype(np.float32)
    else:
        p = np.ones([pa.shape[0], 3], np.float32)
        p[:, 0:2] = pa + xy0
        p = p @ T
    # plots.imshow(im_warped_0//2 + im0_roi//2)
    return p, v


# @profile
def KLTwarp(im, im0, im0_small, p0):
    # Parameters for lucas kanade optical flow
    EPS = cv2.TERM_CRITERIA_EPS
    COUNT = cv2.TERM_CRITERIA_COUNT
    lk_coarse = dict(winSize=(13, 13), maxLevel=5, criteria=(EPS | COUNT, 20, 0.1))
    lk_fine = dict(winSize=(31, 31), maxLevel=1, criteria=(EPS | COUNT, 40, 0.001))

    # 1. Coarse tracking on full image
    scale = 1 / 8
    im_small = cv2.resize(im, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    p, v, _ = cv2.calcOpticalFlowPyrLK(im0_small, im_small, p0 * scale, None, **lk_coarse)
    v = v.ravel() == 1
    p = p / scale
    # plots.imshow(im0_small//2+im_small//2, p1=p0[v]*scale,p2=p[v]*scale)
    T23, inliers = cv2.estimateAffine2D(p0[v], p[v], method=cv2.RANSAC)  # 2x3, better results
    v[v] = inliers.ravel() == 1
    # plots.imshow(im0_small//2+im_small//2, p1=p0[v]*scale,p2=p[v]*scale)

    # 2. Coarse tracking on reduced, translated region https://www.mathworks.com/discovery/affine-transformation.html
    T = np.eye(3, 2)
    T[2,] = (p[v] - p0[v]).mean(0)  # translation-only transform
    p, v = KLTregional(im0, im, p0, T, lk_coarse, fbt=.5, translateFlag=True)

    if v.sum() > 10:  # good fit
        # T23 = cv2.estimateRigidTransform(p0[v], p[v], fullAffine=True)
        T23, inliers = cv2.estimateAffine2D(p0[v], p[v], method=cv2.RANSAC)  # 2x3, better results
    else:
        print('KLT coarse-affine failure, running ORB matches.')
        T23, inliers = orb_match(im0, im, p0)

    # 3. Fine tracking on affine-transformed regions
    p, v = KLTregional(im0, im, p0, T23.T, lk_fine, fbt=0.1)
    return p, v, im_small


# @profile
def estimatePlatePosition(K, p_im, p_w):
    # Linear solution
    # R, t = extrinsicsPlanar(p_im, p_w, K)
    # x0 = np.concatenate([dcm2rpy(R), t])

    # Nonlinear Least Squares
    p_w3 = np.zeros([p_w.shape[0], 3], np.float32)
    p_w3[:, 0:2] = p_w

    x0 = np.array([0, 0, 0, 0, 0, 1], np.float32)
    R, t = fcnNLScamera2world(K, p_im, p_w3, x0)

    # Residuals
    p_im_projected = world2image(K, R, t, p_w3)
    residuals = np.sqrt(((p_im_projected - p_im) ** 2).sum(axis=1))
    return t, R, residuals, p_im_projected


def image2world(K, R, t, p):
    # Copy of MATLAB function pointsToworld
    tform = np.concatenate([R[0:2, :], t[None]]) @ K
    p3 = np.ones([p.shape[0], 3], np.float32)
    p3[:, 0:2] = p
    p_w = p3 @ np.linalg.inv(tform)
    return p_w[:, 0:2] / p_w[:, 2:3]


def world2image(K, R, t, p_w):
    # Copy of MATLAB function worldToImage
    camera_matrix = np.concatenate([R, t[None]]) @ K
    p4 = np.ones([p_w.shape[0], 4], np.float32)
    p4[:, 0:3] = p_w
    p = p4 @ camera_matrix
    return p[:, 0:2] / p[:, 2:3]


# @profile
def extrinsicsPlanar(imagePoints, worldPoints, K):
    # Copy of MATLAB function by same name
    # s[uv1]' = cam_intrinsics * [R t] * [xyz1]'

    # Compute homography.
    # H = fitgeotrans(worldPoints, imagePoints, 'projective')
    H, inliers = cv2.findHomography(worldPoints, imagePoints, method=0)  # methods = 0, RANSAC = 8, LMEDS, RHO
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    A = K.T
    # Ainv = np.linalg.inv(A)
    # b = Ainv @ h1[:, None]
    b = np.linalg.solve(A, h1)  # slower than np.linalg.inv(A) @ h1[:,None]
    lambda_ = 1 / norm3(b)  # lambda_ = 1 / norm(A \ h1) in MATLAB

    # Compute rotation
    r1 = b * lambda_  # r1 = np.linalg.solve(A, lambda_ * h1)
    r2 = np.linalg.solve(A, lambda_ * h2)
    # r1 = Ainv @ (lambda_ * h1[:, None]).ravel()
    # r2 = Ainv @ (lambda_ * h2[:, None]).ravel()

    r3 = np.cross(r1, r2)
    R = np.stack((r1, r2, r3), axis=0)

    # R may not be a true rotation matrix because of noise in the data.
    # Find the best rotation matrix to approximate R using SVD.
    U, _, V = np.linalg.svd(R)
    R = U @ V

    # Compute translation vector
    T = np.linalg.solve(A, lambda_ * h3)
    # T =  Ainv @ (lambda_ * h3[:, None]).ravel()
    return R, T


# @profile
def fcnNLScamera2world(K, p, p_w, x):
    # K = 3x3 intrinsic matrix
    # p = nx2 image points
    # p_w = nx3 world points
    def fzhat(a, K):
        # om = np.ones([n, 4])
        # om[:, 0:3] = a
        # zhat = om @ cam_matrix
        zhat = a @ K  # simplify for special case of R=eye(3), t=[0,0,0]
        return (zhat[:, 0:2] / zhat[:, 2:3]).ravel()

    # R = np.eye(3)
    # t = np.zeros([1, 3])
    # cam_matrix = np.concatenate([R, t]) @ K

    # https://la.mathworks.com/help/vision/ref/cameramatrix.html
    # Using the camera matrix and homogeneous coordinates, you can project a world point onto the image.
    # w * [x,y,1] = [X,Y,Z,1] * camMatrix
    # (X,Y,Z): world coordinates of a point
    # (x,y): coordinates of the corresponding image point
    # w: arbitrary scale factor

    # x = 6 x 1 for 6 parameters
    # J = n x 6
    # z = n x 1 for n measurements

    dx = 1E-6  # for numerical derivatives
    dx1 = [dx, 0, 0]
    dx2 = [0, dx, 0]
    dx3 = [0, 0, dx]
    n = p_w.shape[0]
    z = p.ravel()  # nz=numel(z)
    max_iter = 500
    mdm = np.diag(np.eye(6) * .5)  # marquardt damping matrix (eye times damping coeficient)
    for i in np.arange(max_iter):
        x03 = x[0:3]
        x36 = x[3:6]
        a0 = p_w @ rpy2dcm(x03)
        a1 = p_w @ rpy2dcm(x03 + dx1)
        a2 = p_w @ rpy2dcm(x03 + dx2)
        a3 = p_w @ rpy2dcm(x03 + dx3)
        b0 = x36
        b1 = x36 + dx1
        b2 = x36 + dx2
        b3 = x36 + dx3

        zhat = fzhat(a0 + b0, K)
        J = np.concatenate((fzhat(a1 + b0, K),
                            fzhat(a2 + b0, K),
                            fzhat(a3 + b0, K),
                            fzhat(a0 + b1, K),
                            fzhat(a0 + b2, K),
                            fzhat(a0 + b3, K)), axis=0).reshape(n * 2, 6, order='F')

        J = ((J - zhat[None].T) / dx)
        JTJ = J.T @ J
        delta = np.linalg.inv(JTJ + mdm) @ J.T @ (z - zhat)
        # delta = np.linalg.solve(JTJ + mdm, J.T) @ (z - zhat)  # slower, but possibly more stable??
        x = x + delta
        delta_rms = math.sqrt((delta * delta).sum()) / 6
        if delta_rms < 1E-9:
            break
    if i == (max_iter - 1):
        print('WARNING: fcnNLScamera2world() reaching max iterations!')
    R = rpy2dcm(x[0:3])
    t = x[3:6]
    return R, t


def rpy2dcm(rpy):
    # [roll, pitch, yaw] to direction cosine matrix
    sr = math.sin(rpy[0])
    sp = math.sin(rpy[1])
    sy = math.sin(rpy[2])
    cr = math.cos(rpy[0])
    cp = math.cos(rpy[1])
    cy = math.cos(rpy[2])

    return np.array([[cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
                     [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
                     [- sp, sr * cp, cr * cp]])


def dcm2rpy(DCM):
    # direction cosine matrix to [roll, pitch, yaw] aka [phi, theta, psi]
    rpy = np.zeros(3)
    rpy[0] = math.atan(DCM[2, 1] / DCM[2, 2])
    rpy[1] = math.asin(-DCM[2, 0])
    rpy[2] = math.atan2(DCM[1, 0], DCM[0, 0])
    return rpy

# SCRATCH -------------------------------------------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import time
# from scipy import interpolate
#
# # tform = np.matrix( np.random.rand(3,3)/10 )
# # im_warped = fcnimwarp(im, cam['ixy'], tform)
# # cv2.AffineTransformer.warpImage()
#
# from skimage.transform import (warp, warp_coords, rotate, resize, rescale,
#                                AffineTransform,
#                                ProjectiveTransform,
#                                SimilarityTransform,
#                                downscale_local_mean)
#
# x = np.arange(1920 * 1080 * 4, dtype=np.)
# x = x.reshape(1920 * 2, 1080 * 2)
#
# theta = (- np.pi / 2)
# # tform = AffineTransform(scale=1, rotation=theta, translation=(0, 4))
# tform = SimilarityTransform(scale=1, rotation=theta, translation=(0, 4))
#
# x90 = warp(x, tform, order=1)
# print(x90 - np.rot90(x))
#
# x90 = warp(x, tform.inverse, order=1)
# x90 - np.rot90(x)
#
# imagename = '/Users/glennjocher/Downloads/DATA/VSM/2018.3.11/IMG_4124.JPG'
# im = cv2.imread(imagename, 0)
#
# tform23 = np.array([[0.95719, 0.010894],
#                     [-0.035418, 0.89938],
#                     [-105.34, 49.714]]).T  # 2x3 affine tform
#
# tform33 = np.array([[0.95719, 0.010894, 0],
#                     [-0.035418, 0.89938, 0],
#                     [-105.34, 49.714, 1]]).T  # 2x3 affine tform
#
# tic=time.time(); im_warped1 = cv2.warpAffine(im, tform23, (im.shape[1], im.shape[0])); print(time.time()-tic)
# tic=time.time(); im_warped2 = warp(im, tform33, order=1); print(time.time()-tic)
#
# plt.imshow(im_warped1)
# cv2.remap()
# # cv2.convertMaps()
