# Ultralytics YOLO 🚀, AGPL-3.0 License https://ultralytics.com/license

from utils.common import *
from utils.NLS import fzK


# @profile
def fcnMSV1_t(K, P, B, vg, ii):  # solves for 1 camera translation
    """Solves for 1 camera translation by optimizing translation vectors given camera matrix K, pixel coordinates P,
    baseline B, valid mask vg, and image index ii.
    """
    # vg = np.isnan(P[0, :, i])==False
    nf = ii + 1
    ng = vg.sum()
    U = np.zeros((3, nf, ng))
    for j in range(nf):
        U[:, j] = pixel2uvec(K, P[0:2, vg, j].T).T
    u0 = B[0, 0:3] - B[:nf, 0:3]
    x = np.array([0, 0, 1]) - u0[nf - 2]

    dx = 1e-6  # for numerical derivatives
    dx1, dx2, dx3 = [dx, 0, 0], [0, dx, 0], [0, 0, dx]
    z = P[0:2, vg, ii].ravel("F")
    max_iter = 1000
    mdm = np.eye(3) * 1  # marquardt damping matrix (eye times damping coefficient)
    xi = np.zeros((max_iter, 3))
    res = np.zeros((max_iter, 1))
    for i in range(max_iter):
        b0 = fcn2vintercept(np.vstack((u0[:-1], -x)), U) + x
        zhat = fzK(b0, K).ravel()

        JT = fzK(np.concatenate((b0 + dx1, b0 + dx2, b0 + dx3), 0), K).reshape(3, ng * 2)
        JT = (JT - zhat) / dx  # J Transpose
        JTJ = JT @ JT.T  # J.T @ J

        delta = np.linalg.inv(JTJ + mdm) @ JT @ (z - zhat)  # * min(((i + 1) * .1) ** 2, 1)
        res[i] = rms(z - zhat)
        # print('%g: f=%g, x=%s' % (i, rms(z - zhat), rms(delta)))
        x = x + delta
        xi[i] = x
        if rms(delta) < 1e-8:
            break
    if i == (max_iter - 1):
        print("WARNING: fcnMSV1_t() reaching max iterations!")
    # print('%i steps, residual rms = %.5f' % (i,rms(z-zhat)))
    # py.plot([go.Scatter(x=xi[:i, 0], y=xi[:i, 2], mode='markers',
    #                    marker=dict(
    # size='16', color=(np.log10(res[:i]).ravel()), colorscale='Viridis',showscale=True))])
    return x.astype(np.float32), b0


def fcnMSV2_t(K, P, B, vg, i):  # solves for 1 camera translation
    """Solves for 1 camera translation via iterative minimization, returns optimized camera parameters as np.float32."""
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

    dx = 1e-6  # for numerical derivatives
    dx1, dx2, dx3 = [dx, 0, 0], [0, dx, 0], [0, 0, dx]
    z = P[0:2, vg, i - 1 : i + 1].ravel("F")
    max_iter = 300
    mdm = np.eye(6) * 1  # marquardt damping matrix (eye times damping coefficient)
    for i in range(max_iter):
        a = fcnNvintercept(np.vstack((u0[:-2], -x.reshape((2, 3)))), U)
        a1 = a + x[:3]
        a2 = a + x[3:6]

        zhat = fzK(np.vstack((a1, a2)), K).ravel()
        residual = z - zhat
        JT0 = np.zeros((3, ng * 2))
        JT1 = fzK(np.concatenate((a1 + dx1, a1 + dx2, a1 + dx3), 0), K).reshape(3, ng * 2)
        JT2 = fzK(np.concatenate((a2 + dx1, a2 + dx2, a2 + dx3), 0), K).reshape(3, ng * 2)
        Jtop = np.concatenate((JT1, JT0), 1)
        Jbot = np.concatenate((JT0, JT2), 1)
        JT = np.concatenate((Jtop, Jbot), 0)

        JT = (JT - zhat) / dx
        JTJ = JT @ JT.T  # J.T @ J
        delta = np.linalg.inv(JTJ + mdm) @ JT @ residual * min(((i + 1) * 0.01) ** 2, 1)
        print(f"{i:g}: f={rms(z - zhat):g}, x={rms(delta)}")
        x = x + delta
        if rms(delta) < 1e-8:
            break
    if i == (max_iter - 1):
        print("WARNING: fcnMSV2_t() reaching max iterations!")
    # print('%i steps, residual rms = %.5f' % (i,rms(z-zhat)))
    return x.astype(np.float32)


# @profile
def fcn2vintercept(A, U):
    """Calculates 3D intercepts of vectors (U) from origins (A) for camera frame combinations, returning nx3 tie point
    centers.
    """
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
    """Calculates the vector intercept using frame points and velocities, returning a (n_points, 3) array."""
    _, nf, nv = U.shape  # 3, nframes, npoints
    C0 = np.zeros((nv, 3))

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
        C0[j] = np.linalg.inv(S1[j]) @ S2[:, j]
    return C0


# EXPERIMENTAL ---------------------------------------------------------------------------------------------------------
# EXPERIMENTAL ---------------------------------------------------------------------------------------------------------


def fcnMSV1direct_t(K, P, B, vg, i):  # solves for 1 camera translation
    """Solves for 1 camera translation using minimization of reprojection error through gradient descent."""

    def loss_fn(x, u0, U, K, z):
        b0 = fcn2vintercept(np.vstack((u0[:-1], -x)), U) + x
        zhat = fzK(b0, K)
        return rms(z - zhat)

    nf = i + 1
    ng = vg.sum()
    U = np.zeros((3, nf, ng))
    for j in range(nf):
        U[:, j] = pixel2uvec(K, P[0:2, vg, j].T).T
    u0 = B[0, 0:3] - B[:nf, 0:3]
    x = np.array([0, 0, 1]) - u0[nf - 2]
    Z = P[0:2, vg, i].ravel("F")

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
        dx = 1e-6  # for numerical derivatives
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
        m_cap = m / (1 - (beta_1**i))  # calculates the bias-corrected estimates
        v_cap = v / (1 - (beta_2**i))  # calculates the bias-corrected estimates
        delta = (alpha * m_cap) / (v_cap**0.5 + epsilon)
        x = x - delta  # updates the parameters
        print(f"Residual {r[i]:g},    Params: {x[:]}")
        xi[i] = x
        if rms(delta) < 1e-5:  # convergence check
            break

    py.plot(
        [
            go.Scatter(
                x=xi[:i, 0],
                y=xi[:i, 2],
                mode="markers",
                marker=dict(size="16", color=(np.log10(r[:i]).ravel()), colorscale="Viridis", showscale=True),
            )
        ]
    )

    xhat, _ = fcnMSV1_t(K, P, B, vg, 2)
    uvec(B[1, 0:3] - B[0, 0:3])
    uvec(x)
    uvec(xhat)
