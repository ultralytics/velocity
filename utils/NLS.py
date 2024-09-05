# Ultralytics YOLO 🚀, AGPL-3.0 License https://ultralytics.com/license

import time

from utils.transforms import *


# @profile
def estimateWorldCameraPose(K, p, p3, t=np.array([0, 0, 1]), R=np.eye(3), findR=False):
    """
    Estimates camera pose from world coordinates using non-linear least squares.

    Args: K (array): camera matrix, p (2D array): image points, p3 (3D array): world points, t (array, optional): initial translation, R (array, optional): initial rotation, findR (bool, optional): if True estimates rotation. Returns: tuple (translation, rotation, residuals, projected points).
    """
    # Linear solution
    # Re, te = extrinsicsPlanar(p, p3, K)

    # Nonlinear Least Squares
    x0 = np.concatenate((dcm2rpy(R), t))
    if findR is True:
        R, t = fcnNLS_Rt(K.astype(float), p.astype(float), p3, x0)
    else:
        t = fcnNLS_t(K.astype(float), p.astype(float), p3, t)

    # q = dcm2quat(R)
    # print(q)
    # R2 = quat2dcm(q)

    # Residuals
    p_proj = world2image(K, R, t, p3)
    residuals = rms(p - p_proj)
    return t, R, residuals, p_proj


# @profile
def extrinsicsPlanar(imagePoints, worldPoints, K):  # Copy of MATLAB function by same name
    """Estimates camera extrinsic parameters from planar object points and their image projections."""
    # s[uv1]' = K * [R t] * [xyz1]'

    # Compute homography.
    H, inliers = cv2.findHomography(worldPoints, imagePoints, method=0)  # methods = 0, RANSAC = 8, LMEDS, RHO
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

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


def fzK(a, K):
    """Transforms points `a` using camera matrix `K` and scales; `a` shape (N,2), `K` shape (3,3)."""
    # R = np.eye(3)
    # t = np.zeros([1, 3])
    # camMatrix = np.concatenate([R, t]) @ K
    # zhat = np.concatenate((a,np.ones((a.shape[0],1))), axis=1)  @ camMatrix  # general case
    # zhat = a @ K  # special case of R=eye(3), t=[0,0,0]
    return pscale(a @ K)


def fzC(a, K, R, t=np.zeros((1, 3))):
    """Applies perspective scaling to points `a` using camera matrix from `K`, `R`, and optionally `t`, returning scaled
    points.
    """
    camMatrix = np.concatenate([R, t]) @ K
    return pscale(addcol1(a) @ camMatrix)


def fcnLS_R(K, p, pw):  # MSVM paper EQN 20
    """Solves for rotation matrix `R` from camera matrix `K`, pixel `p`, and world points `pw` using SVD and least
    squares method.
    """
    z = pixel2uvec(K, p)
    H = uvec(pw)
    R = np.linalg.solve(H.T @ H, H.T) @ z
    U, _, V = np.linalg.svd(R)
    R = U @ V
    return R


# @profile
def fcnNLS_t(K, p, pw, x):
    """Estimates translation vector `x` using Non-Linear Least Squares for camera calibration with given `K`, `p`, and
    `pw`.
    """
    # K = 3x3 intrinsic matrix
    # p = nx2 image points
    # pw = nx3 world points = pc @ cam2ned().T

    dx = 1e-6  # for numerical derivatives
    dx1, dx2, dx3 = [dx, 0, 0], [0, dx, 0], [0, 0, dx]
    n = pw.shape[0]
    z = p.ravel()
    max_iter = 30
    mdm = np.eye(3) * 1  # marquardt damping matrix (eye times damping coefficient)
    for i in range(max_iter):
        b0 = pw + x[:4]
        zhat = fzK(b0, K).ravel()
        JT = fzK(np.concatenate((b0 + dx1, b0 + dx2, b0 + dx3), 0), K).reshape(3, n * 2)
        JT = (JT - zhat) / dx  # J Transpose
        JTJ = JT @ JT.T  # J.T @ J
        delta = np.linalg.inv(JTJ + mdm) @ JT @ (z - zhat) * min(((i + 1) * 0.2) ** 2, 1)
        x = x + delta
        if rms(delta) < 1e-8:
            break
    else:
        print("WARNING: fcnNLS_t() reaching max iterations!")
    # print('%i steps, residual rms = %.5f' % (i,rms(z-zhat)))
    return x.astype(np.float32)


# @profile
def fcnNLS_Rt(K, p, pw, x):
    """Estimates camera rotation (R) and translation (t) from world to image points using Non-Linear Least Squares."""
    # K = 3x3 intrinsic matrix
    # p = nx2 image points
    # pw = nx3 world points = pc @ cam2ned().T
    # def fzKautograd_Rt(x, pw, K):  # for autograd
    #     a = pw @ rpy2dcm(x[0:3]) + x[3:6]
    #     return pscale(a @ K)

    # https://la.mathworks.com/help/vision/ref/cameramatrix.html
    # Project a world point X onto the image point x with arbitrary scale factor w
    # w * [x,y,1] = [X,Y,Z,1] * camMatrix
    # x = 6 x 1 for 6 parameters
    # J = n x 6
    # z = n x 1 for n measurements

    dx = 1e-6  # for numerical derivatives
    dx1, dx2, dx3 = [dx, 0, 0], [0, dx, 0], [0, 0, dx]
    n = pw.shape[0]
    z = p.ravel()
    max_iter = 30
    mdm = np.eye(6) * 1  # marquardt damping matrix (eye times damping coefficient)
    # jfunc = jacobian(fzKautograd_Rt)
    for i in range(max_iter):
        x03 = x[:3]
        x36 = x[3:6]
        a0 = pw @ rpy2dcm(x03)
        a1 = pw @ rpy2dcm(x03 + dx1)  # quat2dcm(
        a2 = pw @ rpy2dcm(x03 + dx2)
        a3 = pw @ rpy2dcm(x03 + dx3)
        b0 = x36
        b1 = x36 + dx1
        b2 = x36 + dx2
        b3 = x36 + dx3
        zhat = fzK(a0 + b0, K).ravel()

        # Ja = jfunc(x, pw, K)  # autograd jacobian (super slow!!)
        JT = fzK(np.concatenate((a1 + b0, a2 + b0, a3 + b0, a0 + b1, a0 + b2, a0 + b3), 0), K).reshape(6, n * 2)
        JT = (JT - zhat) / dx  # J Transpose

        JTJ = JT @ JT.T  # J.T @ J
        delta = np.linalg.inv(JTJ + mdm) @ JT @ (z - zhat) * min(((i + 1) * 0.2) ** 2, 1)
        x = x + delta
        if rms(delta) < 1e-8:
            break
    else:
        print("WARNING: fcnNLS_Rt() reaching max iterations!")
    R = rpy2dcm(x[:3]).astype(np.float32)
    t = x[3:6].astype(np.float32)
    # print('%i steps, residual rms = %.5f' % (i, rms(z-zhat)))
    return R, t


def fcnNLS_batch(K, P, pw, cw):  # solves for pxyz, cxyz[1:], crpy[1:]
    """Solves for camera and tiepoint positions and orientations given keypoints and initial estimates; iteratively
    minimizes reprojection errors.
    """
    v = np.isfinite(P[4]).sum(1) == P.shape[2]  # valid tracks ( > 2 frames long)
    P, pw = P[:, v], pw[v]
    _, nt, nc = P.shape  # number of tiepoints, number of cameras
    nc = nc - 1  # do not fit camera 1 R=eye(3), t=[0,0,0]
    nx = nt * 3 + nc * 6  # number of parameters
    nz = nt * (nc + 1) * 2  # number of measurements
    K = K.astype(float)

    z = P[:2].ravel("F")
    z = np.concatenate((z[::2], z[1::2]))
    nanz = np.isnan(z)
    z[nanz] = 0
    crpy = np.zeros((nc, 3))
    x = np.concatenate((pw, cw[1:], crpy)).ravel()  # [tp_pos, cam_pos, cam_rpy, K3]
    norm(cw[1])

    def fzKautograd_batch(x, K, nc, nt):  # for autograd
        pw = x[: nt * 3].reshape(nt, 3)
        alist = [pw]  # = pw @ np.eye(3) + np.zeros((1,3)), camera 1 fixed
        for i in range(nc):
            ia = nt * 3 + i * 3  # pos start
            ib = nc * 3 + ia  # rpy start
            pos = x[ia : ia + 3]
            rpy = x[ib : ib + 3]
            alist.append(pw @ rpy2dcm(rpy) + pos)
        phat = np.asarray(alist).reshape(((nc + 1) * nt, 3))
        return pscale(phat @ K).ravel("F")

    jdx = 1e-6  # for numerical derivatives
    max_iter = 10
    mdm = np.eye(nx) * 1  # marquardt damping matrix (eye times damping coefficient)
    # jfunc = jacobian(fzKautograd_batch)
    for i in range(max_iter):
        tic = time.time()
        zhat = fzKautograd_batch(x, K, nc, nt)
        zhat[nanz] = 0

        # JT = jfunc(x, K, nc, nt, nx).T  # autograd jacobian (super slow!!)
        JT = np.zeros((nx, nz))
        for j in range(nx):
            x1 = x.copy()
            x1[j] += jdx
            JT[j] = fzKautograd_batch(x1, K, nc, nt)
        JT = (JT - zhat) / jdx

        delta = np.linalg.inv(JT @ JT.T + mdm) @ JT @ (z - zhat) * 0.9
        x = x + delta
        # x[nt * 3:nt * 3 + nc * 3] *= range_cal / norm(x[nt * 3:nt * 3 + 3])  # calibrate scale
        print(f"{i:g}: {time.time() - tic:.3f}s, f={rms(z - zhat):g}, x={rms(delta)}")
        if rms(delta) < 1e-7:
            break
    else:
        print("WARNING: fcnNLS_batch() reaching max iterations!")
    print(f"fcnNLS_batch done in {i:g} steps, {time.time() - tic:.3f}s, f={rms(z - zhat):g}")

    j = nt * 3
    pw = x[:j].reshape(nt, 3)
    cw = x[j : j + nc * 3].reshape(nc, 3)  # cam pos
    cw = np.concatenate((np.zeros((1, 3)), cw), 0)
    x[j + nc * 3 : j + nc * 3 * 2].reshape(nc, 3)  # cam rpy
    return cw, pw


def fcnNLS_batch2(K, P, pw, cw):  # solves for pxyz, [el, az, c_ranges[1:]]
    """Solves for camera and tiepoint positions by fitting to projections, returns camera positions and tiepoint
    positions.
    """
    v = np.isfinite(P[4]).sum(1) == P.shape[2]  # valid tracks ( > 2 frames long)
    P, pw = P[:, v], pw[v]
    _, nt, nc = P.shape  # number of tiepoints, number of cameras
    nc = nc - 1  # do not fit camera 1 R=eye(3), t=[0,0,0]
    nx = nt * 3 + nc + 2 + 3  # number of parameters
    nz = nt * (nc + 1) * 2  # number of measurements
    K = K.astype(float)
    C = cam2ned()

    z = P[:2].ravel("F")
    z = np.concatenate((z[::2], z[1::2]))
    nanz = np.isnan(z)
    z[nanz] = 0

    crpy = np.zeros(3)
    sc = cc2sc(C @ (cw[1] - cw[0]))  # cam 2 ned
    ranges = np.arange(1, nc + 1) * sc[0]
    x = np.concatenate((pw.ravel(), crpy, sc[1:3], ranges))  # [tp_pos, cam_jointrpy, elaz, cam_ranges]

    def fzKautograd_batch(x, K, nc, nt):  # for autograd
        j = nt * 3
        R = rpy2dcm(x[j : j + 3])
        pc = x[:j].reshape(nt, 3) @ R
        sc = np.zeros((nc, 3))
        sc[:, 0] = x[j + 5 : j + 5 + nc]  # ranges
        sc[:, 1] = x[j + 3]  # el
        sc[:, 2] = x[j + 4]  # az
        offset = sc2cc(sc) @ C  # ned to cam
        alist = [pc]  # = pw @ np.eye(3) + np.zeros((1,3)), camera 1 fixed
        for i in range(nc):
            alist.append(pc + offset[i])
        phat = np.asarray(alist).reshape(((nc + 1) * nt, 3))
        return pscale(phat @ K).ravel("F")

    jdx = 1e-6  # for numerical derivatives
    max_iter = 20
    mdm = np.eye(nx) * 1  # marquardt damping matrix (eye times damping coefficient)
    # jfunc = jacobian(fzKautograd_batch)
    for i in range(max_iter):
        tic = time.time()
        zhat = fzKautograd_batch(x, K, nc, nt)
        zhat[nanz] = 0

        # JT = jfunc(x, K, nc, nt, nx).T  # autograd jacobian (super slow!!)
        JT = np.zeros((nx, nz))
        for j in range(nx):
            x1 = x.copy()
            x1[j] += jdx
            JT[j] = fzKautograd_batch(x1, K, nc, nt)
        JT = (JT - zhat) / jdx

        delta = np.linalg.inv(JT @ JT.T + mdm) @ JT @ (z - zhat) * 0.9
        x = x + delta
        # calibrate ranges
        # x[nt * 3 + 5:nt * 3 + 5 + nc] = x[nt * 3 + 5:nt * 3 + 5 + nc] / x[nt * 3 + 5:nt * 3 + 6] * ranges[0]
        # print('%g: %.3fs, f=%g, x=%s' % (i, time.time() - tic, rms(z - zhat), rms(delta)))
        if rms(delta) < 1e-7:
            break
    else:
        print("WARNING: fcnNLS_batch() reaching max iterations!")
    print(f"fcnNLS_batch2 done in {i:g} steps, {time.time() - tic:.3f}s, f={rms(z - zhat):g}")

    j = nt * 3
    sc = np.zeros((nc, 3))
    sc[:, 0] = x[j + 5 : j + 5 + nc]  # ranges
    sc[:, 1] = x[j + 3]  # el
    sc[:, 2] = x[j + 4]  # az
    pw = x[:j].reshape(nt, 3)
    cw = sc2cc(sc) @ C  # cam pos
    cw = np.concatenate((np.zeros((1, 3)), cw), 0)
    x[j : j + 3]  # cam rpy
    return cw, pw
