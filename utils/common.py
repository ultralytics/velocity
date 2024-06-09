import math

import numpy as np
import torch

# Set printoptions
torch.set_printoptions(linewidth=320, precision=5, profile="long")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})  # format short g, %precision=5


def norm(x, axis=None):
    """Calculates the L2 norm of array `x` along specified `axis` (default: all elements)."""
    return (x * x).sum(axis) ** 0.5


def rms(x, axis=None):
    """Computes the root mean square of array `x` along specified `axis` (default: all elements)."""
    return (x * x).mean(axis) ** 0.5


def uvec(x, axis=1):  # turns each row or col into a unit vector
    """Normalizes rows or columns of `x` to unit vectors along `axis`=1 (rows default) or 0; retains original shape."""
    return x / (x * x).sum(axis, keepdims=True) ** 0.5


def addcol0(x):  # append a zero column to right side
    """Appends a zero column to the right of array `x`, returning a new array with retained original dtype."""
    y = np.zeros((x.shape[0], x.shape[1] + 1), x.dtype)
    y[:, :-1] = x
    return y


def addcol1(x):  # append a ones column to right side
    """Appends a ones column to the right of array `x`, returning a new array with retained original dtype."""
    y = np.ones((x.shape[0], x.shape[1] + 1), x.dtype)
    y[:, :-1] = x
    return y


def image2world3(R, t, p):  # image coordinate to world coordinate
    """Converts image coordinates to world coordinates using rotation matrix `R`, translation vector `t`, and points
    `p`.
    """
    return addcol1(p) @ R + t


def image2world(K, R, t, p):  # MATLAB pointsToworld copy
    """Converts image coordinates `p` to world coordinates using camera intrinsics `K`, rotation `R`, and translation
    `t`.
    """
    tform = np.concatenate([R[0:2, :], t[None]]) @ K
    pw = addcol1(p) @ np.linalg.inv(tform)
    return pw[:, 0:2] / pw[:, 2:3]


def world2image(K, R, t, pw):  # MATLAB worldToImage copy
    """Converts world coordinates `pw` to image coordinates using camera intrinsics `K`, rotation `R`, and translation
    `t`.
    """
    camMatrix = np.concatenate([R, t[None]]) @ K
    p = addcol1(pw) @ camMatrix  # nx4 * 4x3
    return p[:, 0:2] / p[:, 2:3]


def elaz(x):  # cartesian coordinate to spherical el and az angles
    """Converts Cartesian coordinates `x` to spherical elevation and azimuth angles; input shape can be (3,) or (N,
    3).
    """
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
    """Converts cartesian coordinates to spherical, input shape (N, 3) or (3,), output shape matches input."""
    s = np.zeros_like(x)
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
    """Converts spherical coordinates [range, elevation, azimuth] to cartesian coordinates [x, y, z]."""
    x = np.zeros_like(s)
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
    """Converts pixel coordinates to azimuth and elevation angles using camera matrix `K` and pixel `p`."""
    p = addcol0(p - K[2, 0:2])
    p[:, 2] = K[0, 0]  # focal length (pixels)
    return elaz(p @ cam2ned().T)


def pixel2uvec(K, p):
    """Converts pixel coordinates `p` to unit vectors using camera matrix `K`."""
    p = addcol0(p - K[2, 0:2])
    p[:, 2] = K[0, 0]  # focal length (pixels)
    return uvec(p)


def fcnsigmarejection(x, srl=3.0, ni=3):
    """Applies sigma rejection for outlier removal in `x`, with sigma level `srl` and iterations `ni`, returning
    filtered array and mask.
    """
    v = np.empty_like(x, dtype=bool)
    v[:] = True
    x = x.ravel()
    for _ in range(ni):
        s = x.std() * srl
        mu = x.mean()
        vi = (x < mu + s) & (x > mu - s)
        x = x[vi]
        v[v] = vi
    return x, v


def pscale(p3):  # normalizes camera coordinates so last column = 1
    """Normalizes camera coordinates to make last column 1; input p3 is Nx3, returns Nx2 array."""
    return p3[:, 0:2] / p3[:, 2:3]


def worldPointsLicensePlate(country="EU"):  # Returns x, y coordinates of license plate
    """
    Returns x, y coordinates of a standard license plate for given country; defaults to EU.

    Usage: `worldPointsLicensePlate(country='Chile')`.
    """
    size = [0.3725, 0.1275, 0] if country == "Chile" else [0.520, 0.110, 0]
    return np.array([[1, -1, 0], [1, 1, 0], [-1, 1, 0], [-1, -1, 0]], np.float32) * np.array(size, np.float32) / 2


def cam2ned():  # x_ned(3x5) = R * x_cam(3x5)   - EQUALS -   x_ned(5x3) = x_cam(5x3) * R'
    """Converts camera coordinates to NED (North-East-Down) coordinate system."""
    # +X_ned(NORTH) = +Z_cam(NORTH)
    # +Y_ned(EAST)  = +X_cam(EAST)
    # +Z_ned(DOWN)  = +Y_cam(DOWN)
    return np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # R
