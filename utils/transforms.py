# Ultralytics YOLO 🚀, AGPL-3.0 License https://ultralytics.com/license

from utils.common import *


# @profile
def rpy2dcm(rpy):  # [roll, pitch, yaw] to direction cosine matrix
    """Converts roll, pitch, and yaw angles (rpy) to a direction cosine matrix; input shape [3]."""
    sr, cr = math.sin(rpy[0]), math.cos(rpy[0])
    sp, cp = math.sin(rpy[1]), math.cos(rpy[1])
    sy, cy = math.sin(rpy[2]), math.cos(rpy[2])

    C = np.zeros((3, 3))
    C[0, 0] = cp * cy
    C[0, 1] = sr * sp * cy - cr * sy
    C[0, 2] = cr * sp * cy + sr * sy
    C[1, 0] = cp * sy
    C[1, 1] = sr * sp * sy + cr * cy
    C[1, 2] = cr * sp * sy - sr * cy
    C[2, 0] = -sp
    C[2, 1] = sr * cp
    C[2, 2] = cr * cp
    return C


# @profile
def transform(X, rpy, t):  # transform X = X @ R + t
    """Applies a rotation and translation transform to points `X` using rotation `rpy` and translation `t`."""
    sr, cr = math.sin(rpy[0]), math.cos(rpy[0])
    sp, cp = math.sin(rpy[1]), math.cos(rpy[1])
    sy, cy = math.sin(rpy[2]), math.cos(rpy[2])
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]

    C00 = cp * cy
    C01 = sr * sp * cy - cr * sy
    C02 = cr * sp * cy + sr * sy
    C10 = cp * sy
    C11 = sr * sp * sy + cr * cy
    C12 = cr * sp * sy - sr * cy
    C20 = -sp
    C21 = sr * cp
    C22 = cr * cp

    return np.concatenate(
        (C00 * x + C10 * y + C20 * z + t[0], C01 * x + C11 * y + C21 * z + t[1], C02 * x + C12 * y + C22 * z + t[2])
    ).reshape((x.size, 3), order="F")


def dcm2rpy(R):  # direction cosine matrix to [roll, pitch, yaw] aka [phi, theta, psi]
    """Converts direction cosine matrix to roll, pitch, and yaw (`[phi, theta, psi]`)."""
    rpy = np.zeros(3)
    rpy[0] = math.atan(R[2, 1] / R[2, 2])
    rpy[1] = math.asin(-R[2, 0])
    rpy[2] = math.atan2(R[1, 0], R[0, 0])
    return rpy


def quat2dcm(q):  # 3 element quaternion representation (roll is norm(q))
    """Converts a quaternion to direction cosine matrix; expects `q` as [x, y, z, w], returns 3x3 matrix."""
    r = norm(q)  # (0.5 - 1.5) for (-180 to +180 deg roll)
    rpy = [(r - 10), math.asin(-q[2] / r), math.atan(q[1] / q[0])]
    return rpy2dcm(rpy)


def dcm2quat(R):  # 3 element quaternion representation (roll is norm(q))
    """
    Converts direction cosine matrix `R` to quaternion; returns quaternion as [x, y, z, w].

    Expects `R` as a 3x3 matrix.
    """
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
