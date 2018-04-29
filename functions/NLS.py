# from functions.common import *
from functions.transforms import *


# @profile
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


# @profile
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


def fzK(a, K):
    # R = np.eye(3)
    # t = np.zeros([1, 3])
    # camMatrix = np.concatenate([R, t]) @ K
    # zhat = np.concatenate((a,np.ones((a.shape[0],1))), axis=1)  @ camMatrix  # general case
    # zhat = a @ K  # special case of R=eye(3), t=[0,0,0]
    return pscale(a @ K)


def fzC(a, K, R, t=np.zeros((1, 3))):
    camMatrix = np.concatenate([R, t]) @ K
    return pscale(addcol1(a) @ camMatrix)


def fcnLS_R(K, p, pw):  # MSVM paper EQN 20
    z = pixel2uvec(K, p)
    H = uvec(pw)
    R = np.linalg.solve(H.T @ H, H.T) @ z
    U, _, V = np.linalg.svd(R)
    R = U @ V
    return R


# @profile
def fcnNLS_t(K, p, pw, x):
    # K = 3x3 intrinsic matrix
    # p = nx2 image points
    # pw = nx3 world points = pc @ cam2ned().T

    dx = 1E-6  # for numerical derivatives
    dx1, dx2, dx3 = [dx, 0, 0], [0, dx, 0], [0, 0, dx]
    n = pw.shape[0]
    z = p.ravel()
    max_iter = 100
    mdm = np.eye(3) * 1  # marquardt damping matrix (eye times damping coeficient)
    for i in range(max_iter):
        b0 = pw + x[0:4]
        zhat = fzK(b0, K).ravel()
        JT = fzK(np.concatenate((b0 + dx1, b0 + dx2, b0 + dx3), 0), K).reshape(3, n * 2)
        JT = (JT - zhat) / dx  # J Transpose
        JTJ = JT @ JT.T  # J.T @ J
        delta = np.linalg.inv(JTJ + mdm) @ JT @ (z - zhat) * min(((i + 1) * .2) ** 2, 1)
        # print('Residual %g,    Params: %s' % (rms(z-zhat), x[:]))

        x = x + delta
        if rms(delta) < 1E-8:
            break
    if i == (max_iter - 1):
        print('WARNING: fcnNLS_t() reaching max iterations!')
    # print('%i steps, residual rms = %.5f' % (i,rms(z-zhat)))
    return x.astype(np.float32)


# @profile
def fcnNLS_Rt(K, p, pw, x):
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

    dx = 1E-6  # for numerical derivatives
    dx1, dx2, dx3 = [dx, 0, 0], [0, dx, 0], [0, 0, dx]
    n = pw.shape[0]
    z = p.ravel()
    max_iter = 100
    mdm = np.eye(6) * 1  # marquardt damping matrix (eye times damping coeficient)
    # jfunc = jacobian(fzKautograd_Rt)
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
        zhat = fzK(a0 + b0, K).ravel()

        # Ja = jfunc(x, pw, K)  # autograd jacobian (super slow!!)
        JT = fzK(np.concatenate((a1 + b0, a2 + b0, a3 + b0, a0 + b1, a0 + b2, a0 + b3), 0), K).reshape(6, n * 2)
        JT = (JT - zhat) / dx  # J Transpose

        JTJ = JT @ JT.T  # J.T @ J
        delta = np.linalg.inv(JTJ + mdm) @ JT @ (z - zhat) * min(((i + 1) * .2) ** 2, 1)
        #         print('Residual %g,    Params: %s' % (rms(z-zhat), x[:]))
        x = x + delta
        if rms(delta) < 1E-8:
            break
    if i == (max_iter - 1):
        print('WARNING: fcnNLS_Rt() reaching max iterations!')
    R = rpy2dcm(x[0:3]).astype(np.float32)
    t = x[3:6].astype(np.float32)
    # print('%i steps, residual rms = %.5f' % (i, rms(z-zhat)))
    return R, t


# @profile
def fcnNLS_batch(K, P, pw, cw):
    v = np.isfinite(P[4]).sum(1) == P.shape[2]  # valid tracks ( > 2 frames long)
    P, pw = P[:, v], pw[v]
    _, nt, nc = P.shape  # number of tiepoints, number of cameras
    nc = nc - 1  # do not fit camera 1 R=eye(3), t=[0,0,0]
    nx = nt * 3 + nc * 6  # number of parameters
    nz = nt * (nc + 1) * 2  # number of measurements
    K = K.astype(float)

    z = P[0:2].ravel('F')
    z = np.concatenate((z[0::2], z[1::2]))
    nanz = np.isnan(z)
    z[nanz] = 0

    crpy = np.zeros((nc, 3))
    x = np.concatenate((pw, cw[1:], crpy)).ravel()  # [tp_pos, cam_pos, cam_rpy, K3]

    # @profile
    def fzKautograd_batch(x, K, nc, nt):  # for autograd
        pw = x[0:nt * 3].reshape(nt, 3)

        # zhat = np.zeros((0, 3))
        alist = [pw]  # = pw @ np.eye(3) + np.zeros((1,3)), camera 1 fixed
        for i in range(nc):
            ia = nt * 3 + i * 3  # pos start
            ib = nc * 3 + ia  # rpy start
            pos = x[ia:ia + 3]
            rpy = x[ib:ib + 3]
            pnew = pw @ rpy2dcm(rpy) + pos
            # zhat = np.concatenate((zhat, pnew))
            alist.append(pnew)
        zhat = np.asarray(alist).reshape(((nc + 1) * nt, 3))
        return pscale(zhat @ K).ravel('F')

    jdx = 1E-6  # for numerical derivatives
    max_iter = 10
    mdm = np.eye(nx) * 1  # marquardt damping matrix (eye times damping coeficient)
    # jfunc = jacobian(fzKautograd_batch)
    # ov, zv = np.ones(nt), np.zeros(nt)
    for i in range(max_iter - 1):
        tic = time.time()
        # pw = x[0:nt * 3].reshape(nt, 3)  # tp pos
        # cw = x[nt * 3:nt*3 + nc*3].reshape(nc, 3)  # camera pos
        # ca = x[nt * 3+nc*3:nt * 3 + nc * 3*2].reshape(nc, 3)  # camera rpy

        # zhat = np.zeros((0, 3))
        # for j in range(nc):
        #     ia = nt * 3 + j * 3  # pos start
        #     ib = nc * 3 + ia  # rpy start
        #     ic = nt * 2 * j  # tp start
        #     pos = x[ia:ia + 3]
        #     rpy = x[ib:ib + 3]
        #     zhat = np.concatenate((zhat, pw @ rpy2dcm(rpy) + pos))
        #
        #     f = K[0, 0]  # focal length (pixels)
        #     dx = pos[0] - pw[:, 0]
        #     dy = pos[1] - pw[:, 1]
        #     dz = pos[2] - pw[:, 2]
        #     sr = np.sin(rpy[0])
        #     sp = np.sin(rpy[1])
        #     sy = np.sin(rpy[2])
        #     cr = np.cos(rpy[0])
        #     cp = np.cos(rpy[1])
        #     cy = np.cos(rpy[2])
        #     k1 = cp * cy * dx - sp * dz + cp * sy * dy
        #     k2 = cr * sy - cy * sp * sr
        #     k3 = cr * cy + sp * sr * sy
        #     k4 = sr * sy + cr * cy * sp
        #     k5 = cy * sr - cr * sp * sy
        #     k6 = cp * sr * dz
        #     k7 = cp * cr * dz
        #     f0 = f / k1
        #     f1 = f / (k1 * k1)
        #
        #     dzdtxyz = np.concatenate((
        #         k2 * f0 + cp * cy * (k3 * dy - k2 * dx + k6) * f1,
        #         cp * cy * (k4 * dx - k5 * dy + k7) * f1 - k4 * f0,
        #         cp * sy * (k3 * dy - k2 * dx + k6) * f1 - k3 * f0,
        #         k5 * f0 + cp * sy * (k4 * dx - k5 * dy + k7) * f1,
        #         - cp * sr * f0 - sp * (k3 * dy - k2 * dx + k6) * f1,
        #         - cp * cr * f0 - sp * (k4 * dx - k5 * dy + k7) * f1
        #     )).reshape((3, nt * 2))
        #
        #     dzdcrpy = np.concatenate((
        #         (k4 * dx - k5 * dy + k7) * f0,
        #         - (k3 * dy - k2 * dx + k6) * f0,
        #         (cp * cy * sr * dx - sp * sr * dz + cp * sr * sy * dy) * f0 +
        #         (k3 * dy - k2 * dx + k6) * (cp * dz + cy * sp * dx + sp * sy * dy) * f1,
        #         (cp * cr * cy * dx - cr * sp * dz + cp * cr * sy * dy) * f0 +
        #         (k4 * dx - k5 * dy + k7) * (cp * dz + cy * sp * dx + sp * sy * dy) * f1,
        #         - (k3 * dx + k2 * dy) * f0 - (cp * cy * dy - cp * sy * dx) * (k3 * dy - k2 * dx + k6) * f1,
        #         (k5 * dx + k4 * dy) * f0 - (cp * cy * dy - cp * sy * dx) * (k4 * dx - k5 * dy + k7) * f1
        #     )).reshape((3, nt * 2))
        #
        #     dzdK = np.concatenate((
        #         ov, zv,
        #         zv, ov,
        #         (k3 * dy - k2 * dx + k6) / k1, (k4 * dx - k5 * dy + k7) / k1
        #     )).reshape((3, nt * 2))
        #
        #     rows = np.mod(np.arange(nt * 3 * 2, dtype=int), nt * 3)
        #     cols = np.floor(np.arange(0, nt * 2, 1 / 3)).astype(int)
        #     JT[rows, cols] = dzdtxyz.ravel()
        #     JT[ia:ia + 3, ic:ic + nt * 2] = -dzdtxyz  # dzdcxyz
        #     JT[ib:ib + 3, ic:ic + nt * 2] = dzdcrpy
        #     JT[nx - 3:nx, ic:ic + nt * 2] = dzdK
        # zhat = fzK(zhat, K).ravel('F')

        zhat = fzKautograd_batch(x, K, nc, nt)
        zhat[nanz] = 0

        JT = np.zeros((nx, nz))
        # JT = jfunc(x, K, nc, nt, nx).T  # autograd jacobian (super slow!!)
        for j in range(nx):
            x1 = x.copy()
            x1[j] += jdx
            JT[j] = (fzKautograd_batch(x1, K, nc, nt) - zhat) / jdx

        JTJ = JT @ JT.T  # J.T @ J
        delta = np.linalg.inv(JTJ + mdm) @ JT @ (z - zhat) * .8
        print('%g: %.3fs, f=%g, x=%s' % (i, time.time() - tic, rms(z - zhat), rms(delta)))
        x = x + delta
        if rms(delta) < 1E-7:
            break
    if i == max_iter:
        print('WARNING: fcnNLS_batch() reaching max iterations!')
    # print('%i steps, residual rms = %.5f' % (i,rms(z-zhat)))

    pw = x[0:nt * 3].reshape(nt, 3)  # tp pos
    cw = x[nt * 3:nt * 3 + nc * 3].reshape(nc, 3)  # camera pos
    ca = x[nt * 3 + nc * 3:nt * 3 + nc * 3 * 2].reshape(nc, 3)  # camera rpy
    print(ca)

    cw = np.concatenate((np.zeros((1, 3)), cw), 0)
    return cw, pw
