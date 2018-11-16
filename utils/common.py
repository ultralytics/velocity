import math
import cv2  # pip install opencv-python opencv-contrib-python
import numpy as np
import scipy.io
# import tensorflow as tf
import torch
import time

# Set printoptions
torch.set_printoptions(linewidth=1320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5


# np.set_string_function(lambda a: str(a.shape), repr=False)

# import autograd.numpy as np  # pip install autograd
# from autograd import jacobian


def mean(x, axis=None):
    if axis is None:
        return x.sum() / x.size
    else:
        return x.sum(axis) / x.shape[axis]


def norm(x, axis=None):
    if axis is None:
        return math.sqrt((x * x).sum())
    else:
        return (x * x).sum(axis) ** 0.5


def rms(x, axis=None):
    if axis is None:
        return math.sqrt((x * x).sum() / x.size)
    else:
        return ((x * x).sum(axis) / x.shape[axis]) ** 0.5


def uvec(x, axis=1):  # turns each row or col into a unit vector
    if axis is None:
        return x / math.sqrt((x * x).sum())
    else:
        return x / (x * x).sum(axis, keepdims=True) ** 0.5


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
    s = np.zeros(x.shape)
    if x.shape[0] == 3:  # by columns
        r = norm(x, axis=0)
        s[0] = r
        s[1] = np.arcsin(-x[2] / r)
        s[2] = np.arctan2(x[1], x[0])
    else:  # by rows
        r = norm(x, axis=1)
        s[:, 0] = r
        s[:, 1] = np.arcsin(-x[:, 2] / r)
        s[:, 2] = np.arctan2(x[:, 1], x[:, 0])
    return s


def sc2cc(s):  # spherical to cartesian [range, el, az] to [x, y z]
    x = np.zeros(s.shape)
    if s.shape[0] == 3:  # by columns
        r = s[0]
        a = r * np.cos(s[1])
        x[0] = a * np.cos(s[2])
        x[1] = a * np.sin(s[2])
        x[2] = -r * np.sin(s[1])
    else:  # by rows
        r = s[:, 0]
        a = r * np.cos(s[:, 1])
        x[:, 0] = a * np.cos(s[:, 2])
        x[:, 1] = a * np.sin(s[:, 2])
        x[:, 2] = -r * np.sin(s[:, 1])
    return x


def pixel2angle(K, p):
    p = addcol0(p - K[2, 0:2])
    p[:, 2] = K[0, 0]  # focal length (pixels)
    return elaz(p @ cam2ned().T)


def pixel2uvec(K, p):
    p = addcol0(p - K[2, 0:2])
    p[:, 2] = K[0, 0]  # focal length (pixels)
    return uvec(p)


def fcnsigmarejection(x, srl=3.0, ni=3):
    v = np.empty_like(x, dtype=bool)
    v[:] = True
    x = x.ravel()
    for m in range(ni):
        s = x.std() * srl
        mu = mean(x)
        vi = (x < mu + s) & (x > mu - s)
        x = x[vi]
        v[v] = vi
    return x, v


def pscale(p3):  # normalizes camera coordinates so last column = 1
    return p3[:, 0:2] / p3[:, 2:3]


def worldPointsLicensePlate():  # Returns x, y coordinates of license plate
    # https://en.wikipedia.org/wiki/Vehicle_registration_plate
    size = [.3725, .1275, 0]  # [0.36 0.13] (m) license plate size (Chile)
    return np.array([[1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]], np.float32) * (np.array(size, np.float32) / 2)


def cam2ned():  # x_ned(3x5) = R * x_cam(3x5)   - EQUALS -   x_ned(5x3) = x_cam(5x3) * R'
    # +X_ned(NORTH) = +Z_cam(NORTH)
    # +Y_ned(EAST)  = +X_cam(EAST)
    # +Z_ned(DOWN)  = +Y_cam(DOWN)
    return np.array([[0, 0, 1],
                     [1, 0, 0],
                     [0, 1, 0]])  # R
