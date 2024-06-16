# Ultralytics YOLO 🚀, AGPL-3.0 License https://ultralytics.com/license

import cv2

from utils.common import *
from utils.strings import *


def boundingRect(x, imshape, border=(0, 0)):
    """Calculates bounded rectangle for `x` with optional border, adjusting to `imshape` limits; x (contours), imshape
    (shape), border ((w,h)).
    """
    x0, y0, width, height = cv2.boundingRect(x)
    x0, y0, x1, y1 = x0 - border[0], y0 - border[1], x0 + width + border[0], y0 + height + border[1]
    x0 = max(x0, 1)
    y0 = max(y0, 1)
    x1 = min(x1, imshape[1])
    y1 = min(y1, imshape[0])
    return x0, x1, y0, y1


def insidebbox(x, box):
    """Checks if points in `x` are within `box`, returning a bool array; `x` shape is (N,2), `box` is (x0,x1,y0,y1)."""
    x0, x1, y0, y1 = box
    v = np.zeros(x.shape[0], bool)
    v[(x[:, 0] > x0) & (x[:, 0] < x1) & (x[:, 1] > y0) & (x[:, 1] < y1)] = True
    return v


def importEXIF(fullfilename):
    """Parses EXIF data from an image file specified by `fullfilename`, returning a dict with processed EXIF values."""
    import exifread

    exif = exifread.process_file(open(fullfilename, "rb"), details=False)
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


def fcnEXIF2LLAT(E):  # E = image exif info i.e. E = importEXIF('img.jpg')
    """Extracts latitude, longitude, altitude, and timestamp from image EXIF data as [lat, long, alt, time]."""
    # llat = [lat, long, alt (m), time (s)]
    # MATLAB:  datenum('2018:03:11 15:57:22','yyyy:mm:dd HH:MM:SS') # fractional day since 00/00/000
    # Python:  d = datetime.strptime('2018:03:11 15:57:22', "%Y:%m:%d %H:%M:%S"); datetime.toordinal(d) + 366
    # day = datenum(E['EXIF DateTimeOriginal'] + '.' + E['EXIF SubsecTimeOriginal'], 'yyyy:mm:dd HH:MM:SS.FFF')

    s = E["EXIF DateTimeOriginal"]
    s = s.split(" ")[1]
    hour, minute, second = s.split(":")
    day_fraction = (
        float(hour) / 24 + float(minute) / 1440 + float(second) / 86400 + E["EXIF SubSecTimeOriginal"] / 86400000
    )
    # d = datetime.strptime(E['EXIF DateTimeOriginal'], "%Y:%m:%d %H:%M:%S")
    # day = datetime.toordinal(d) + 366
    # day_fraction = d.hour / 24 + d.minute / 1440 + d.second / 86400 + E['EXIF SubSecTimeOriginal'] / 86400000

    llat = np.zeros(4)
    llat[0] = dms2degrees(E["GPS GPSLatitude"]) * hemisphere2sign(E["GPS GPSLatitudeRef"])
    llat[1] = dms2degrees(E["GPS GPSLongitude"]) * hemisphere2sign(E["GPS GPSLongitudeRef"])
    llat[2] = E["GPS GPSAltitude"]
    llat[3] = day_fraction * 86400  # seconds since midnight
    return llat


def dms2degrees(dms):  # maps GPS [degrees minutes seconds] to decimal degrees
    """Converts GPS [degrees minutes seconds] to decimal degrees; `dms` is a list of [degrees, minutes, seconds]."""
    return dms[0] + dms[1] / 60 + dms[2] / 3600


def hemisphere2sign(x):  # converts hemisphere strings 'N', 'S', 'E', 'W' to signs 1, -1, 1, -1
    """Converts hemisphere strings ('N', 'S', 'E', 'W') to signs (1, -1) respectively; `x` is an array of hemisphere
    characters.
    """
    sign = np.zeros(len(x))
    sign[(x == "N") | (x == "E")] = 1
    sign[(x == "S") | (x == "W")] = -1
    return sign


# # @profile
def getCameraParams(fullfilename, platform="iPhone 6s"):  # returns camera parameters and file information structure cam
    """Extracts camera parameters and info structure for images/videos from a given file path, supports iPhone 6s
    platform.
    """
    pathname, _, extension, filename = filenamesplit(fullfilename)
    isvideo = (extension == ".MOV") | (extension == ".mov") | (extension == ".m4v")

    if platform == "iPhone 6s":
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
            diagonalRatio = math.sqrt(4032**2 + 3024**2) / math.sqrt(3840**2 + 2160**2)

            focalLength_pix = np.array([3486, 3486]) * diagonalRatio

            orientation = 1 if width > height else 6
        else:  # 12MP IMAGE 4032x3024
            cap = []
            exif = importEXIF(fullfilename)
            kltBlockSize = [21, 21]

            orientation = exif["Image Orientation"]
            width = exif["EXIF ExifImageWidth"]
            height = exif["EXIF ExifImageLength"]
            fps = 0
            frame_count = 1

            focalLength_pix = [3486, 3486]
            # focalLength_pix = exif['EXIF FocalLength'] / sensorSize(1) * width
        skew = 0
    elif platform == "iPhone x":
        "fill in here"

    radialDistortion = [0, 0, 0]
    principalPoint = np.array([width, height]) / 2 + 0.5

    # focalLength_pix = [3548.9, 3670.8]  # from MATLAB
    # principalPoint = [1909.1, 1125.5]  # from MATLAB

    IntrinsicMatrix = np.array(
        [[focalLength_pix[0], 0, 0], [skew, focalLength_pix[1], 0], [principalPoint[0], principalPoint[1], 1]],
        np.float32,
    )

    if orientation == 1:  # 1 = landscape, 6 = vertical
        orientation_comment = "Horizontal"
    elif orientation == 6:
        orientation_comment = "Vertical"

    cam = {
        "fullfilename": fullfilename,
        "pathname": pathname,
        "filename": filename,
        "extension": extension,
        "isvideo": isvideo,
        "width": width,
        "height": height,
        "sensorSize_mm": sensorSize_mm,
        "focalLength_mm": focalLength_mm,
        "focalLength_pix": focalLength_pix,
        "fov": fov,
        "skew": skew,
        "principalPoint": principalPoint,
        "IntrinsicMatrix": IntrinsicMatrix,
        "radialDistortion": radialDistortion,
        "ixy": None,
        "kltBlockSize": kltBlockSize,
        "orientation": orientation,
        "orientation_comment": orientation_comment,
        "fps": fps,
        "frame_count": frame_count,
    }
    return cam, cap
