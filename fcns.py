import math
import cv2  # pip install opencv-python, pip install opencv-contrib-python
import numpy as np
# import tensorflow as tf
# import torch
# import plotly.offline as py
# import plotly.graph_objs as go
import plots

np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5


# import autograd.numpy as np
# from autograd import jacobian

# Set options
# pd.set_option('display.width', desired_width)
# np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
#  np.set_printoptions(linewidth=320, formatter={'float_kind': '{:21.15g}'.format})  # format long g, %precision=15
# %precision '%0.8g'


def rms(x):
    return math.sqrt((x * x).sum()) / x.size


def norm(x):
    s = x.shape
    if len(s) > 1:
        n = s[1]
        if n == 3:
            return (x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2) ** 0.5
        elif n == 2:
            return (x[:, 0] ** 2 + x[:, 1] ** 2) ** 0.5
        else:
            return (x * x).sum(axis=1) ** 0.5
    else:
        return math.sqrt((x * x).sum())


def uvec(x, axis=1):  # turns each row or col into a unit vector
    if x.shape.__len__() == 1:
        return x / (x * x).sum() ** 0.5
    else:
        r = (x * x).sum(axis=axis) ** 0.5
        return x / r[:, None]


def worldPointsLicensePlate():  # Returns x, y coordinates of license plate
    # https://en.wikipedia.org/wiki/Vehicle_registration_plate
    size = [.3725, .1275, 0]  # [0.36 0.13] (m) license plate size (Chile)
    return np.array([[1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]], np.float32) * (np.array(size, np.float32) / 2)


def elaz(x):  # cartesian coordinate to spherical el and az angles
    s = x.shape
    r = norm(x)
    if len(s) == 1:
        ea = np.array([math.asin(-x[2] / r), math.atan2(x[1], x[0])])
    else:
        ea = np.zeros((s[0], 2))
        ea[:, 0] = np.arcsin(-x[:, 2] / r)
        ea[:, 1] = np.arctan2(x[:, 1], x[:, 0])
    return ea


def cc2sc(x):
    s = np.zeros_like(x)
    r = norm(x)
    s[0] = r
    s[1] = np.arcsin(-x[2] / r)
    s[2] = np.arctan2(x[1], x[0])
    return s


def sc2cc(s):  # spherical to cartesian [range, el, az] to [x, y z]
    x = np.zeros_like(s)
    r = s[0]
    k1 = r * np.cos(s[1])
    x[0] = k1 * np.cos(s[2])
    x[1] = k1 * np.sin(s[2])
    x[2] = -r * np.sin(s[1])
    return x


def pixel2angle(K, p):
    p = addcol0(p - K[2, 0:2])
    p[:, 2] = K[0, 0]  # focal length (pixels)
    return elaz(p @ cam2ned().T)


def pixel2uvec(K, p):
    p = addcol0(p - K[2, 0:2])
    p[:, 2] = K[0, 0]  # focal length (pixels)
    return uvec(p)


def cam2ned():  # x_ned(3x5) = R * x_cam(3x5)   - EQUALS -   x_ned(5x3) = x_cam(5x3) * R'
    # +X_ned(NORTH) = +Z_cam(NORTH)
    # +Y_ned(EAST)  = +X_cam(EAST)
    # +Z_ned(DOWN)  = +Y_cam(DOWN)
    return np.array([[0, 0, 1],
                     [1, 0, 0],
                     [0, 1, 0]])  # R


def fcnEXIF2LLAT(E):  # E = image exif info i.e. E = importEXIF('img.jpg')
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


def dms2degrees(dms):  # maps GPS [degrees minutes seconds] to decimal degrees
    return dms[0] + dms[1] / 60 + dms[2] / 3600


def hemisphere2sign(x):  # converts hemisphere strings 'N', 'S', 'E', 'W' to signs 1, -1, 1, -1
    sign = np.zeros_like(x)
    sign[(x == 'N') | (x == 'E')] = 1
    sign[(x == 'S') | (x == 'W')] = -1
    return sign


def filenamesplit(string):  # splits a full filename string into path, file, and extension.
    # Example:  path, file, extension, fileext = filenamesplit('/Users/glennjocher/Downloads/IMG_4124.JPG')
    i = string.rfind('/') + 1
    j = string.rfind('.')
    path, file, extension = string[:i], string[i:j], string[j:]
    return path, file, extension, file + extension


# # @profile
def getCameraParams(fullfilename, platform='iPhone 6s'):  # returns camera parameters and file information structure cam
    pathname, _, extension, filename = filenamesplit(fullfilename)
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
            frame_count = 1

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
        'ixy': None,
        'kltBlockSize': kltBlockSize,
        'orientation': orientation,
        'orientation_comment': orientation_comment,
        'fps': fps,
        'frame_count': frame_count}
    return cam, cap


def importEXIF(fullfilename):
    import exifread
    exif = exifread.process_file(open(fullfilename, 'rb'), details=False)
    for tag in exif.keys():
        a = exif[tag].values[:]
        if type(a) is str and a.isnumeric():
            a = float(a)
        if type(a) is list:
            n = a.__len__()
            a = np.asarray(a)
            for i in range(n):
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


# # @profile
def boundingRect(x, imshape, border=(0, 0)):
    x0, y0, width, height = cv2.boundingRect(x)
    x0, y0, x1, y1 = x0 - border[0], y0 - border[1], x0 + width + border[0], y0 + height + border[1]
    if x0 < 1: x0 = 1
    if y0 < 1: y0 = 1
    if x1 > imshape[1]:  x1 = imshape[1]
    if y1 > imshape[0]:  y1 = imshape[0]
    return x0, x1, y0, y1


def insidebbox(x, box):
    x0, x1, y0, y1 = box
    v = np.zeros(x.shape[0], bool)
    v[(x[:, 0] > x0) & (x[:, 0] < x1) & (x[:, 1] > y0) & (x[:, 1] < y1)] = True
    return v


# # @profile
def estimateAffine2D_SURF(im1, im2, p1, scale=1.0):
    im1 = cv2.resize(im1, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    im2 = cv2.resize(im2, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # orb = cv2.AKAZE_create()
    orb = cv2.xfeatures2d.SURF_create()
    bf = cv2.BFMatcher()  # cv2.NORM_HAMMING, crossCheck=True)
    kp2, des2 = orb.detectAndCompute(im2, mask=None)

    a = 0
    ngood = 0
    while ngood < 10:
        border = int(a * scale)
        x0, x1, y0, y1 = boundingRect(p1 * scale, im1.shape, border=(border, border))
        kp1, des1 = orb.detectAndCompute(im1[y0:y1, x0:x1], mask=None)
        matches = bf.knnMatch(des1, des2, k=2)
        # matches = sorted(matches, key=lambda x: x.distance)
        good = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good.append(m)
        ngood = len(good)
        a += 10

    m1 = np.float32([kp1[x.queryIdx].pt for x in good]) + np.float32([x0, y0])
    m2 = np.float32([kp2[x.trainIdx].pt for x in good])
    # plots.imshow(im1 // 2 + im2 // 2, None, m1, m2)
    return cv2.estimateAffine2D(m1 / scale, m2 / scale, method=cv2.RANSAC)  # 2x3, better results


# @profile
def KLTregional(im0, im, p0, T, lk_param, fbt=1.0, translateFlag=False):
    T = T.astype(np.float32)
    # 1. Warp current image to past image frame
    # im_warped_0 = cv2.warpAffine(im, T23, (int(im.shape[1]/2), int(im.shape[0]/2)),flags=cv2.WARP_INVERSE_MAP)
    x0, x1, y0, y1 = boundingRect(p0, im.shape, border=(50, 50))
    im0_roi = im0[y0:y1, x0:x1]
    xy0 = np.float32([x0, y0])
    p0_roi = p0 - xy0

    if translateFlag:
        dx = T[2, 0].__int__()
        dy = T[2, 1].__int__()
        im_warped_0 = im[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    else:
        x, y = np.meshgrid(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32), copy=False)
        x__ = x * T[0, 0] + y * T[1, 0] + T[2, 0]
        y__ = x * T[0, 1] + y * T[1, 1] + T[2, 1]
        im_warped_0 = cv2.remap(im, x__, y__, cv2.INTER_LINEAR)  # current image ROI mapped to previous image

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # im0_roi = clahe.apply(im0_roi)
    # im_warped_0 = clahe.apply(im_warped_0)

    # im0_roi = cv2.equalizeHist(im0_roi)
    # im_warped_0 = cv2.equalizeHist(im_warped_0)

    # run klt tracker forward-backward
    # _, pyr1 = cv2.buildOpticalFlowPyramid(im0_roi, winSize=lk_param['winSize'], maxLevel=lk_param['maxLevel'], withDerivatives=True)
    pa, va, _ = cv2.calcOpticalFlowPyrLK(im0_roi, im_warped_0, p0_roi, None, **lk_param)
    pb, vb, _ = cv2.calcOpticalFlowPyrLK(im_warped_0, im0_roi, pa, None, **lk_param)
    fbe = norm(pb - p0_roi)
    v = (va.ravel() == 1) & (vb.ravel() == 1) & (fbe < fbt)  # forward-backward error threshold

    # convert p back to im coordinates
    if translateFlag:
        p = pa + (xy0 + [dx, dy]).astype(np.float32)
    else:
        p = addcol1(pa + xy0) @ T

    # residuals = norm(p0_roi[v] - pa[v])
    # _, i = fcnsigmarejection(residuals, srl=3, ni=3)
    # v[v] = i
    # plots.imshow(im_warped_0 // 2 + im0_roi // 2, None, p0_roi, pa)
    return p, v


# @profile
def KLTmain(im, im0, im0_small, p0):
    # Parameters for KLT
    EPS = cv2.TERM_CRITERIA_EPS
    COUNT = cv2.TERM_CRITERIA_COUNT
    lk_coarse = dict(winSize=(15, 15), maxLevel=11, criteria=(EPS | COUNT, 20, 0.1))
    lk_fine = dict(winSize=(51, 51), maxLevel=1, criteria=(EPS | COUNT, 40, 0.001))

    # 1. Coarse tracking on 1/8 scale full image
    scale = 1 / 4
    im_small = cv2.resize(im, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    if im0_small is None:
        im0_small = cv2.resize(im0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    p, v, _ = cv2.calcOpticalFlowPyrLK(im0_small, im_small, p0 * scale, None, **lk_coarse)
    v = v.ravel() == 1
    p /= scale
    T23, inliers = cv2.estimateAffine2D(p0[v], p[v], method=cv2.RANSAC)  # 2x3, better results
    v[v] = inliers.ravel() == 1
    # import plots; plots.imshow(im0_small//2+im_small//2, p1=p0[v]*scale,p2=p[v]*scale)

    # 2. Coarse tracking on full resolution roi https://www.mathworks.com/discovery/affine-transformation.html
    translation = p[v] - p0[v]
    T = np.eye(3, 2)
    T[2] = translation.mean(0)  # translation-only transform
    p, v = KLTregional(im0, im, p0, T, lk_coarse, fbt=1, translateFlag=True)

    if v.sum() > 10:  # good fit
        T23, inliers = cv2.estimateAffine2D(p0[v], p[v], method=cv2.RANSAC)  # 2x3, better results
    else:
        print('KLT coarse-affine failure, running SURF matches full scale.')
        T23, inliers = estimateAffine2D_SURF(im0, im, p0, scale=1)

    # 3. Fine tracking on affine-transformed regions
    p, v = KLTregional(im0, im, p0, T23.T, lk_fine, fbt=0.3)
    return p[v], v, im_small


# # @profile
def estimateWorldCameraPose(K, p, p3, t=np.array([0, 0, 1]), R=np.eye(3), findR=False):
    # Linear solution
    # Re, te = extrinsicsPlanar(p, p3, K)

    # Nonlinear Least Squares
    x0 = np.concatenate([dcm2rpy(R), t])
    if findR is True:
        R, t = fcnNLS_Rt(K.astype(float), p.astype(float), p3, x0)
    else:
        t = fcnNLS_t(K.astype(float), p.astype(float), p3, t)

    # q = dcm2quat(R)
    # print(q)
    # R2 = quat2dcm(q)

    # Residuals
    p_proj = world2image(K, R, t, p3)
    residuals = norm(p - p_proj)
    return t, R, residuals, p_proj


def addcol0(x):  # append a zero column to right side
    y = np.zeros((x.shape[0], x.shape[1] + 1), x.dtype)
    y[:, :-1] = x
    return y


def addcol1(x):  # append a ones column to right side
    y = np.ones((x.shape[0], x.shape[1] + 1), x.dtype)
    y[:, :-1] = x
    return y


def image2world3(R, t, p):  # image coordinate to world coordinate
    return addcol1(p) @ R + t


def image2world(K, R, t, p):  # MATLAB pointsToworld copy
    tform = np.concatenate([R[0:2, :], t[None]]) @ K
    pw = addcol1(p) @ np.linalg.inv(tform)
    return pw[:, 0:2] / pw[:, 2:3]


def world2image(K, R, t, pw):  # MATLAB worldToImage copy
    camMatrix = np.concatenate([R, t[None]]) @ K
    p = addcol1(pw) @ camMatrix  # nx4 * 4x3
    return p[:, 0:2] / p[:, 2:3]


# # @profile
def extrinsicsPlanar(imagePoints, worldPoints, K):  # Copy of MATLAB function by same name
    # s[uv1]' = K * [R t] * [xyz1]'

    # Compute homography.
    H, inliers = cv2.findHomography(worldPoints, imagePoints, method=0)  # methods = 0, RANSAC = 8, LMEDS, RHO
    h1 = H[:, 0]
    h2 = H[:, 1]
    h3 = H[:, 2]

    A = K.T
    # Ainv = np.linalg.inv(A)
    # b = Ainv @ h1[:, None]
    b = np.linalg.solve(A, h1)  # slower than np.linalg.inv(A) @ h1[:,None]
    lambda_ = 1 / norm(b)  # lambda_ = 1 / norm(A \ h1) in MATLAB

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
    t = np.linalg.solve(A, lambda_ * h3)
    # t =  Ainv @ (lambda_ * h3[:, None]).ravel()
    return R, t


def pscale(p3):  # normalizes camera coordinates so last column = 1
    return p3[:, 0:2] / p3[:, 2:3]


def fzK(a, K):
    # R = np.eye(3)
    # t = np.zeros([1, 3])
    # camMatrix = np.concatenate([R, t]) @ K
    # zhat = np.concatenate((a,np.ones((a.shape[0],1))), axis=1)  @ camMatrix  # general case
    # zhat = a @ K  # special case of R=eye(3), t=[0,0,0]
    return pscale(a @ K).ravel()


def fzC(a, K, R, t=np.zeros((1, 3))):
    camMatrix = np.concatenate([R, t]) @ K
    return pscale(addcol1(a) @ camMatrix).ravel()


def fcnMSV1direct_t(K, P, B, vg, i):  # solves for 1 camera translation
    from scipy.optimize import minimize
    def loss_fn(x, u0, U, K, z):
        b0 = fcn2vintercept(np.vstack((u0[:-1], -x)), U) + x
        zhat = fzK(b0, K)
        return ((z - zhat) ** 2).mean() ** 0.5

    nf = i + 1
    ng = vg.sum()
    U = np.zeros((3, nf, ng))
    for j in range(nf):
        U[:, j] = pixel2uvec(K, P[0:2, vg, j].T).T
    u0 = B[0, 0:3] - B[:nf, 0:3]
    x = np.array([0, 0, 1]) - u0[nf - 2]
    Z = P[0:2, vg, i].ravel('F')

    x0 = np.array([0, 0, 1])

    # res = minimize(loss_fn, x0, args=(u0, U, K, Z), jac=None, method='Powell')

    # x1 = np.arange(-1, 1, 0.02) * .5
    # y1 = np.arange(-1, 1, 0.02) * .5
    # x, z = np.meshgrid(x1, y1)
    # x0s = x.shape
    # x = x.ravel()
    # z = z.ravel()
    # f = np.zeros_like(x)
    # for i in range(x.size):
    #     xi = np.array([x[i], 0, z[i]])
    #     f[i] = loss_fn(xi, u0, U, K, Z)
    # f = f.reshape(x0s)
    #
    # s, _ = fcnsigmarejection(f, 3, 6)
    # data = go.Contour(z=f, x=x1, y=y1, colorscale='Viridis', ncontours=100,
    #                   autocontour=False,
    #                   contours=dict(start=0, end=s.max(), size=3, coloring='fill'))
    # layout = go.Layout(scene=dict(xaxis=dict(range=[-1, 1], ), yaxis=dict(range=[-1, 1], ), ))
    # fig = go.Figure(data=[data], layout=layout)
    # py.plot(fig)

    # # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = torch.optim.Adam([x, u0, U, K, Z], lr=0.0001)
    #
    # def closure():
    #     optimizer.zero_grad()
    #     # output = model(input)
    #     # loss = loss_fn(output, target)
    #     loss = loss_fn(x, u0, U, K, Z)
    #     loss.backward()
    #     return loss
    # optimizer.step(closure)

    def grad_func(x, u0, U, K, z):  # calculates the gradient
        dx = 1E-6  # for numerical derivatives
        dx1, dx2, dx3 = [dx, 0, 0], [0, dx, 0], [0, 0, dx]
        b0 = fcn2vintercept(np.vstack((u0[:-1], -x)), U) + x
        f0 = rms(z - fzK(b0, K))
        f1 = rms(z - fzK(b0 + dx1, K))
        f2 = rms(z - fzK(b0 + dx2, K))
        f3 = rms(z - fzK(b0 + dx3, K))
        g = (np.array([f1, f2, f3]) - f0) / dx

        # x03 = x[0:3]
        # x36 = x[3:6]
        # U[:, 1] = rpy2dcm(x36).T @ U[:, 1]
        # b0 = fcn2vintercept(np.vstack((u0[:-1], -x03)), U) + x03
        # f0 = rms(z - fzC(b0, K,rpy2dcm(x36).T))
        # f1 = rms(z - fzC(b0 + dx1, K, rpy2dcm(x36).T))
        # f2 = rms(z - fzC(b0 + dx2, K, rpy2dcm(x36).T))
        # f3 = rms(z - fzC(b0 + dx3, K, rpy2dcm(x36).T))
        # f4 = rms(z - fzC(b0, K, rpy2dcm(x36 + dx1).T))
        # f5 = rms(z - fzC(b0, K, rpy2dcm(x36 + dx2).T))
        # f6 = rms(z - fzC(b0, K, rpy2dcm(x36 + dx3).T))
        # g = (np.array([f1, f2, f3, f4, f5, f6]) - f0) / dx
        return g, f0

    alpha = 0.01  # 0.01, learning rate
    beta_1 = 0.9  # 0.9
    beta_2 = 0.999  # 0.999
    epsilon = 1e-8

    x0 = np.array([0, 0, 1])
    x = x0
    m = 0
    v = 0
    max_iter = 2000
    xi = np.zeros((max_iter, 3))
    r = np.zeros((max_iter, 1))
    for i in range(1, max_iter):
        g, r[i] = grad_func(x, u0, U, K, Z)  # computes the gradient of the stochastic function
        m = beta_1 * m + (1 - beta_1) * g  # updates the moving averages of the gradient
        v = beta_2 * v + (1 - beta_2) * (g * g)  # updates the moving averages of the squared gradient
        m_cap = m / (1 - (beta_1 ** i))  # calculates the bias-corrected estimates
        v_cap = v / (1 - (beta_2 ** i))  # calculates the bias-corrected estimates
        delta = (alpha * m_cap) / (v_cap ** 0.5 + epsilon)
        x = x - delta  # updates the parameters
        print('Residual %g,    Params: %s' % (r[i], x[:]))
        xi[i] = x
        if np.mean(delta ** 2) ** 0.5 < 1E-5:  # convergence check
            break

    py.plot([go.Scatter(x=xi[:i, 0], y=xi[:i, 2], mode='markers',
                        marker=dict(size='16', color=(np.log10(r[:i]).ravel()), colorscale='Viridis', showscale=True))])

    xhat, _ = fcnMSV1_t(K, P, B, vg, 2)
    uvec(B[1, 0:3] - B[0, 0:3])
    uvec(x)
    uvec(xhat)


# @profile
def fcnMSV1_t(K, P, B, vg, ii):  # solves for 1 camera translation
    # vg = np.isnan(P[0, :, i])==False
    nf = ii + 1
    ng = vg.sum()
    U = np.zeros((3, nf, ng))
    for j in range(nf):
        U[:, j] = pixel2uvec(K, P[0:2, vg, j].T).T
    u0 = B[0, 0:3] - B[:nf, 0:3]
    x = np.array([0, 0, 1]) - u0[nf - 2]

    dx = 1E-6  # for numerical derivatives
    dx1, dx2, dx3 = [dx, 0, 0], [0, dx, 0], [0, 0, dx]
    z = P[0:2, vg, ii].ravel('F')
    max_iter = 1000
    mdm = np.eye(3) * 1  # marquardt damping matrix (eye times damping coeficient)
    xi = np.zeros((max_iter, 3))
    res = np.zeros((max_iter, 1))
    for i in range(max_iter):
        b0 = fcn2vintercept(np.vstack((u0[:-1], -x)), U) + x
        zhat = fzK(b0, K)
        JT = np.concatenate((fzK(b0 + dx1, K), fzK(b0 + dx2, K), fzK(b0 + dx3, K))).reshape(3, ng * 2)  # J Transpose

        JT = (JT - zhat) / dx
        JTJ = JT @ JT.T  # J.T @ J

        delta = np.linalg.inv(JTJ + mdm) @ JT @ (z - zhat)  # * min(((i + 1) * .1) ** 2, 1)
        res[i] = rms(z - zhat)
        # print('Residual %g,    Params: %s' % ((z - zhat).mean(), x[:]))
        x = x + delta
        xi[i] = x
        if rms(delta) < 1E-9:
            break
    if i == (max_iter - 1):
        print('WARNING: fcnMSV1_t() reaching max iterations!')
    # print('%i steps, residual rms = %.5f' % (i,(z - zhat).mean()))
    # py.plot([go.Scatter(x=xi[:i, 0], y=xi[:i, 2], mode='markers',
    #                    marker=dict(size='16', color=(np.log10(res[:i]).ravel()), colorscale='Viridis',showscale=True))])
    return x.astype(np.float32), b0


def fcnMSV2_t(K, P, B, vg, i):  # solves for 1 camera translation
    # vg = np.isnan(P[0, :, i])==False
    nf = i + 1
    ng = vg.sum()
    U = np.zeros((3, nf, ng))
    for j in range(nf):
        U[:, j] = pixel2uvec(K, P[0:2, vg, j].T).T
    u0 = B[0, 0:3] - B[:nf, 0:3]
    # x = np.array([[0, 0, 1] - u0[nf - 3], [0, 0, 2] - u0[nf - 3]]).ravel()
    # x = -np.array([u0[nf - 2], u0[nf - 2] - [0, 0, 1]]).ravel()
    x = -u0[1:].ravel()

    dx = 1E-6  # for numerical derivatives
    dx1, dx2, dx3 = [dx, 0, 0], [0, dx, 0], [0, 0, dx]
    z = P[0:2, vg, i - 1:i + 1].ravel('F')
    max_iter = 300
    mdm = np.eye(6) * 1  # marquardt damping matrix (eye times damping coeficient)
    for i in range(max_iter):
        a = fcnNvintercept(np.vstack((u0[:-2], -x.reshape((2, 3)))), U)
        a1 = a + x[0:3]
        a2 = a + x[3:6]

        zhat = fzK(np.vstack((a1, a2)), K)
        residual = z - zhat
        JT0 = np.zeros((3, ng * 2))
        JT1 = np.concatenate((fzK(a1 + dx1, K), fzK(a1 + dx2, K), fzK(a1 + dx3, K))).reshape(3, ng * 2)
        JT2 = np.concatenate((fzK(a2 + dx1, K), fzK(a2 + dx2, K), fzK(a2 + dx3, K))).reshape(3, ng * 2)
        Jtop = np.concatenate((JT1, JT0), 1)
        Jbot = np.concatenate((JT0, JT2), 1)
        JT = np.concatenate((Jtop, Jbot), 0)

        JT = (JT - zhat) / dx
        JTJ = JT @ JT.T  # J.T @ J
        delta = np.linalg.inv(JTJ + mdm) @ JT @ residual * min(((i + 1) * .01) ** 2, 1)
        print('Residual %g,    Params: %s' % (residual.mean(), x[:]))
        x = x + delta
        if rms(delta) < 1E-9:
            break
    if i == (max_iter - 1):
        print('WARNING: fcnMSV2_t() reaching max iterations!')
    # print('%i steps, residual rms = %.5f' % (i,(z - zhat).mean()))
    return x.astype(np.float32)


def fcnLS_R(K, p, pw):  # MSVM paper EQN 20
    z = pixel2uvec(K, p)
    H = uvec(pw)
    R = np.linalg.solve(H.T @ H, H.T) @ z
    U, _, V = np.linalg.svd(R)
    R = U @ V
    return R


def fcnNLS_t(K, p, pw, x):
    # K = 3x3 intrinsic matrix
    # p = nx2 image points
    # pw = nx3 world points = pc @ cam2ned().T

    dx = 1E-6  # for numerical derivatives
    dx1, dx2, dx3 = [dx, 0, 0], [0, dx, 0], [0, 0, dx]
    n = pw.shape[0]
    z = p.ravel()  # nz=numel(z)
    max_iter = 100
    mdm = np.eye(3) * 1  # marquardt damping matrix (eye times damping coeficient)
    for i in range(max_iter):
        b0 = pw + x[0:4]
        zhat = fzK(b0, K)
        JT = np.concatenate((fzK(b0 + dx1, K),
                             fzK(b0 + dx2, K),
                             fzK(b0 + dx3, K))).reshape(3, n * 2)  # J Transpose
        JT = (JT - zhat) / dx
        JTJ = JT @ JT.T  # J.T @ J
        delta = np.linalg.inv(JTJ + mdm) @ JT @ (z - zhat) * min(((i + 1) * .2) ** 2, 1)
        # print('Residual %g,    Params: %s' % ((z - zhat).mean(), x[:]))
        x = x + delta
        if rms(delta) < 1E-9:
            break
    if i == (max_iter - 1):
        print('WARNING: fcnNLS_t() reaching max iterations!')
    # print('%i steps, residual rms = %.5f' % (i,(z - zhat).mean()))
    return x.astype(np.float32)


# # @profile
def fcnNLS_Rt(K, p, pw, x):
    # K = 3x3 intrinsic matrix
    # p = nx2 image points
    # pw = nx3 world points = pc @ cam2ned().T

    # def fzhat0(x, pw, K):  # for autograd
    #     zhat = (pw @ rpy2dcm(x[0:3]) + x[3:6]) @ K
    #     return (zhat[:, 0:2] / zhat[:, 2:3]).ravel()

    # https://la.mathworks.com/help/vision/ref/cameramatrix.html
    # Project a world point X onto the image point x with arbitrary scale factor w
    # w * [x,y,1] = [X,Y,Z,1] * camMatrix
    # x = 6 x 1 for 6 parameters
    # J = n x 6
    # z = n x 1 for n measurements

    dx = 1E-6  # for numerical derivatives
    dx1, dx2, dx3 = [dx, 0, 0], [0, dx, 0], [0, 0, dx]
    n = pw.shape[0]
    z = p.ravel()  # nz=numel(z)
    max_iter = 100
    mdm = np.eye(6) * 1  # marquardt damping matrix (eye times damping coeficient)
    # jfunc = jacobian(fzhat0)
    for i in range(max_iter):
        x03 = x[0:3]
        x36 = x[3:6]
        a0 = pw @ rpy2dcm(x03)
        a1 = pw @ rpy2dcm(x03 + dx1)  # quat2dcm(
        a2 = pw @ rpy2dcm(x03 + dx2)
        a3 = pw @ rpy2dcm(x03 + dx3)
        b0 = x36
        b1 = x36 + dx1
        b2 = x36 + dx2
        b3 = x36 + dx3
        zhat = fzK(a0 + b0, K)

        # Ja = jfunc(x, pw, K)  # autograd jacobian (super slow!!)
        JT = np.concatenate((fzK(a1 + b0, K),
                             fzK(a2 + b0, K),
                             fzK(a3 + b0, K),
                             fzK(a0 + b1, K),
                             fzK(a0 + b2, K),
                             fzK(a0 + b3, K))).reshape(6, n * 2)  # J Transpose
        JT = (JT - zhat) / dx

        JTJ = JT @ JT.T  # J.T @ J
        delta = np.linalg.inv(JTJ + mdm) @ JT @ (z - zhat) * min(((i + 1) * .2) ** 2, 1)
        #         print('Residual %g,    Params: %s' % ((z - zhat).mean(), x[:]))
        x = x + delta
        if rms(delta) < 1E-9:
            break
    if i == (max_iter - 1):
        print('WARNING: fcnNLS_Rt() reaching max iterations!')
    R = rpy2dcm(x[0:3])
    t = x[3:6]
    # print('%i steps, residual rms = %.5f' % (i, (z - zhat).mean()))
    return R.astype(np.float32), t.astype(np.float32)


def rpy2dcm(rpy):  # [roll, pitch, yaw] to direction cosine matrix
    sr = math.sin(rpy[0])
    sp = math.sin(rpy[1])
    sy = math.sin(rpy[2])
    cr = math.cos(rpy[0])
    cp = math.cos(rpy[1])
    cy = math.cos(rpy[2])
    return np.array([[cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
                     [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
                     [- sp, sr * cp, cr * cp]])


def dcm2rpy(R):  # direction cosine matrix to [roll, pitch, yaw] aka [phi, theta, psi]
    rpy = np.zeros(3)
    rpy[0] = math.atan(R[2, 1] / R[2, 2])
    rpy[1] = math.asin(-R[2, 0])
    rpy[2] = math.atan2(R[1, 0], R[0, 0])
    return rpy


def quat2dcm(q):  # 3 element quaternion representation (roll is norm(q))
    r = norm(q)  # (0.5 - 1.5) for (-180 to +180 deg roll)
    rpy = [(r - 10), math.asin(-q[2] / r), math.atan(q[1] / q[0])]
    return rpy2dcm(rpy)


def dcm2quat(R):  # 3 element quaternion representation (roll is norm(q))
    r, p, y = dcm2rpy(R)
    return sc2cc(np.array([r + 10, p, y]))


# def quat2dcm(q):  # verified against MATLAB
#     s, x, y, z = q / norm(q)
#     return np.array([1 - 2 * (y * y + z * z), 2 * (x * y - s * z), 2 * (x * z + s * y),
#                      2 * (x * y + s * z), 1 - 2 * (x * x + z * z), 2 * (y * z - s * x),
#                      2 * (x * z - s * y), 2 * (y * z + s * x), 1 - 2 * (x * x + y * y)]).reshape(3, 3)
#
#
# def dcm2quat(R):  # incomplete
#     q = np.zeros(4)
#     return q


def fcnsigmarejection(x, srl=3.0, ni=3):
    v = np.empty_like(x, dtype=bool)
    v[:] = True
    x = x.ravel()
    for m in range(ni):
        s = x.std() * srl
        mu = x.mean()
        vi = (x < mu + s) & (x > mu - s)
        x = x[vi]
        v[v] = vi
    return x, v


# @profile
def fcn2vintercept(A, U):
    # A = nx3 camera origins, ux1 = nxnp x unit vectors
    _, nf, nv = U.shape  # 3, nframes, npoints
    C0 = np.zeros([nv, 3])

    import itertools
    comb = np.array(list(itertools.combinations(range(nf), 2)))
    j = comb[:, 0]
    k = comb[:, 1]

    dA = A[j] - A[k]
    BAx = dA[:, 0:1]
    BAy = dA[:, 1:2]
    BAz = dA[:, 2:3]

    # COMBINATIONS
    vx = U[0, k]
    vy = U[1, k]
    vz = U[2, k]
    ux = U[0, j]
    uy = U[1, j]
    uz = U[2, j]

    # VECTOR INTERCEPTS
    d = ux * vx + uy * vy + uz * vz
    e = ux * BAx + uy * BAy + uz * BAz
    f = vx * BAx + vy * BAy + vz * BAz
    g = 1 - d * d
    s1 = (d * f - e) / g  # multiply times U
    t1 = (f - d * e) / g  # multiply times v

    # MISCLOSURE VECTOR RANGE RESIDUALS
    # r = ((t1 * vx - BAx - s1 * ux) ** 2 + (t1 * vy - BAy - s1 * uy) ** 2 + (t1 * vz - BAz - s1 * uz) ** 2) ** 0.5

    # TIE POINT CENTERS
    den = j.size * 2  # denominator = number of permutations times 2
    B = A.sum(0) * (nf - 1)
    C0[:, 0] = ((t1 * vx + s1 * ux).sum(0) + B[0]) / den
    C0[:, 1] = ((t1 * vy + s1 * uy).sum(0) + B[1]) / den
    C0[:, 2] = ((t1 * vz + s1 * uz).sum(0) + B[2]) / den
    return C0


# @profile
def fcnNvintercept(A, U):
    _, nf, nv = U.shape  # 3, nframes, npoints
    C0 = np.zeros([nv, 3])

    ux1, uy1, uz1 = U[0], U[1], U[2]

    V = np.zeros((9, nf, nv))
    V[0] = 1 - ux1 * ux1
    V[1] = -ux1 * uy1
    V[2] = -ux1 * uz1
    V[3] = V[1]
    V[4] = 1 - uy1 * uy1
    V[5] = -uy1 * uz1
    V[6] = V[2]
    V[7] = V[5]
    V[8] = 1 - uz1 * uz1
    S1 = V.sum(1).T.reshape((nv, 3, 3))

    S2 = np.zeros([3, nv])
    Ax = A[:, 0:1].T
    Ay = A[:, 1:2].T
    Az = A[:, 2:3].T
    S2[0] = Ax @ V[0] + Ay @ V[1] + Az @ V[2]
    S2[1] = Ax @ V[3] + Ay @ V[4] + Az @ V[5]
    S2[2] = Ax @ V[6] + Ay @ V[7] + Az @ V[8]

    for j in range(nv):
        # C0[j] = np.linalg.solve(S1[j], S2[:, j])
        C0[j] = np.linalg.inv(S1[j]) @ S2[:, j]
    return C0


def fcnNvinterceptOrig(A, U):
    _, nf, nv = U.shape  # 3, nframes, npoints
    C0 = np.zeros([nv, 3])

    ux1, uy1, uz1 = U[0], U[1], U[2]
    v1 = 1 - ux1 * ux1
    v2 = -ux1 * uy1
    v3 = -ux1 * uz1
    v4 = v2
    v5 = 1 - uy1 * uy1
    v6 = -uy1 * uz1
    v7 = v3
    v8 = v6
    v9 = 1 - uz1 * uz1

    S1mat = np.zeros([9, nv])
    S1mat[0] = v1.sum(0)
    S1mat[1] = v2.sum(0)
    S1mat[2] = v3.sum(0)
    S1mat[3] = v4.sum(0)
    S1mat[4] = v5.sum(0)
    S1mat[5] = v6.sum(0)
    S1mat[6] = v7.sum(0)
    S1mat[7] = v8.sum(0)
    S1mat[8] = v9.sum(0)

    S2mat = np.zeros([3, nv])
    Ax = A[:, 0:1].T
    Ay = A[:, 1:2].T
    Az = A[:, 2:3].T
    S2mat[0] = Ax @ v1 + Ay @ v2 + Az @ v3
    S2mat[1] = Ax @ v4 + Ay @ v5 + Az @ v6
    S2mat[2] = Ax @ v7 + Ay @ v8 + Az @ v9

    S1m = S1mat.T.reshape((nv, 3, 3))
    for j in range(nv):
        # C0[j, :] = np.linalg.solve(S1m[:, :, j], S2mat[:, j])
        C0[j] = np.linalg.inv(S1m[j]) @ S2mat[:, j]
    return C0


def adamOptimizerExample():
    alpha = 0.5  # 0.01, learning rate
    beta_1 = 0.09  # 0.9
    beta_2 = 0.999  # 0.999
    epsilon = 1e-8

    def func(x):
        return x * x - 4 * x + 4

    def grad_func(x):  # calculates the gradient
        return 2 * x - 4

    x = 0
    m = 0
    v = 0
    xi = np.zeros(300)
    for i in range(1, 300):
        g = grad_func(x)  # computes the gradient of the stochastic function
        m = beta_1 * m + (1 - beta_1) * g  # updates the moving averages of the gradient
        v = beta_2 * v + (1 - beta_2) * (g * g)  # updates the moving averages of the squared gradient
        m_cap = m / (1 - (beta_1 ** i))  # calculates the bias-corrected estimates
        v_cap = v / (1 - (beta_2 ** i))  # calculates the bias-corrected estimates
        delta = (alpha * m_cap) / (math.sqrt(v_cap) + epsilon)
        x = x - delta  # updates the parameters
        xi[i] = x
        if np.mean(delta ** 2) ** 0.5 < 1E-9:  # convergence check
            break
    print(x)
    py.plot([go.Scatter(y=xi)])
