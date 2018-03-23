import math
from scipy import interpolate
import cv2
import numpy as np
# import pandas as pd
import exifread

# Set options
# pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:5.8g}'.format})  # , precision=3, suppress=True)


# %precision '%0.8g'

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
                     [-x, -y]])  # worldPoints


def cam2NED():
    # x_ned(3x5) = R * x_cam(3x5)   - EQUALS -   x_ned(5x3) = x_cam(5x3) * R'
    # +X_ned(NORTH) = +Z_cam(NORTH)
    # +Y_ned(EAST)  = +X_cam(EAST)
    # +Z_ned(DOWN)  = +Y_cam(DOWN)
    return np.array([[0, 0, 1],
                     [1, 0, 0],
                     [0, 1, 0]])  # R


def fcnEXIF2LLAT(E):
    from datetime import datetime
    # E = image exif info i.e. E = importEXIF('img.jpg')
    # llat = [lat, long, alt (m), time (s)]
    # MATLAB:  datenum('2018:03:11 15:57:22','yyyy:mm:dd HH:MM:SS') # fractional day since 00/00/000
    # Python:  d = datetime.strptime('2018:03:11 15:57:22', "%Y:%m:%d %H:%M:%S"); datetime.toordinal(d) + 366
    # day = datenum(E['EXIF DateTimeOriginal'] + '.' + E['EXIF SubsecTimeOriginal'], 'yyyy:mm:dd HH:MM:SS.FFF')

    d = datetime.strptime(E['EXIF DateTimeOriginal'], "%Y:%m:%d %H:%M:%S")
    day = datetime.toordinal(d) + 366
    day_fraction = d.hour / 24 + d.minute / 1440 + d.second / 86400 + E['EXIF SubSecTimeOriginal'] / 86400000

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


def filenamesplit(str):
    # splits a full filename string into path, filename, and extension. Example:
    # str = '/Users/glennjocher/Downloads/DATA/VSM/2018.3.11/IMG_4124.JPG'
    # path = '/Users/glennjocher/Downloads/DATA/VSM/2018.3.11/'
    # filename = 'IMG_4124.JPG'
    # extension = '.JPG'
    i = str.rfind('/') + 1
    path = str[0:i]
    filename = str[i:None]
    extension = filename[filename.rfind('.'):None]
    return path, filename, extension


def getCameraParams(fullfilename, platform='iPhone 6s'):
    # returns camera parameters and file information structure cam
    # fullfilename: video or image(s) file name(s) i.e. mymovie.mov or IMG_3797.jpg
    # platform: camera name i.e. 'iPhone 6s'
    pathname, filename, extension = filenamesplit(fullfilename)
    isvideo = (extension == '.MOV') | (extension == '.mov')

    if platform == 'iPhone 6s':
        # pixelSize = 0.0011905 #(mm) on a side, 12um
        sensorSize_mm = np.array([4.80, 3.60])  # (mm) CMOS sensor
        focalLength_mm = 4.15  # (mm) iPhone 6s from EXIF
        # focalLength_pix= focalLength_mm / sensorSize_mm[0] * width
        # fov = np.degrees([math.atan(width/2/focalLength_pix) math.atan(height/2/focalLength_pix)]*2) # (deg) camera FOV
        fov = np.degrees(np.arctan(sensorSize_mm / 2 / focalLength_mm) * 2)  # (deg) camea field of view

        if isvideo:  # 4k VIDEO 3840x2160
            cap = cv2.VideoCapture(fullfilename)
            # V = VideoReader(fullfilename)
            kltBlockSize = [51, 51]

            # https://docs.opencv.org/3.1.0/d8/dfe/classcv_1_1VideoCapture.html#aeb1644641842e6b104f244f049648f94
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            # ratio of image to video frame diagonal lengths:
            # https://photo.stackexchange.com/questions/86075/does-the-iphones-focal-length-differ-when-taking-video-vs-photos
            diagonalRatio = np.linalg.norm([4032, 3024]) / np.linalg.norm([3840, 2160])

            skew = 0
            focalLength_pix = np.array([3486, 3486]) * diagonalRatio
            cap.release()  # Release the video capture object
        else:  # 12MP IMAGE 4032x3024
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
    IntrinsicMatrix = np.matrix([[focalLength_pix[0], 0, 0],
                                 [skew, focalLength_pix[1], 0],
                                 [principalPoint[0], principalPoint[1], 1]])

    if width > height:  # 1 = landscape, 6 = vertical
        orientation = 1
        orientation_comment = 'Horizontal'
    else:
        orientation = 6
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
    return cam


def importEXIF(fullfilename):
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


def fcnimwarp(I, ixy, tform):
    # warps input image I into output image J according to 3x3 tform and nx3 flattened meshgrid indices ixy
    p = ixy * tform
    pz = p[:, 2]
    py = np.asarray(p[:, 1] / pz)
    px = np.asarray(p[:, 0] / pz)
    # px = ixy*tform[:,0]
    # py = ixy*tform[:,1]  # faster than p=Ixy*tform if no normalization needed (affine only)

    px = px.ravel()
    py = py.ravel()
    xa = np.arange(I.shape[0])
    ya = np.arange(I.shape[1])
    f1 = interpolate.RectBivariateSpline(xa, ya, I)
    J = f1(px, py, dx=1, dy=1, grid=False)

    if I.size == J.size:  # reshape
        J = J.reshape(I.shape)
    return J


def KLTwarp(KLT, Ixy, I, Im1, pm1, vm1, mode):
    # # Create a mask image for drawing purposes
    # mask = np.zeros_like(old_frame)
    # while (1):
    #     ret, frame = cap.read()
    #     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    #     # calculate optical flow
    #     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    #
    #     # Select good points
    #     good_new = p1[st == 1]
    #     good_old = p0[st == 1]
    #
    #     # Now update the previous frame and previous points
    #     old_gray = frame_gray.copy()
    #     p0 = good_new.reshape(-1, 1, 2)

    return None


def make_rand_photons(photonsum):
    mu = 0
    sigma = 1
    s = np.random.normal(mu, sigma, photonsum)
    t = np.random.normal(mu, sigma, photonsum)
    u = np.random.normal(mu, sigma, photonsum)
    d = np.stack((s, t, u), axis=1)
    rangee = np.linalg.norm(d, axis=1)
    d = d / rangee[:, None]
    return d

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
# imagename = '/Users/glennjocher/Downloads/DATA/VSM/2018.3.11/IMG_4124.JPG';
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
# # cv2.remap()
# # cv2.convertMaps()
