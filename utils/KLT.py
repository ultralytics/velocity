# Ultralytics YOLO 🚀, AGPL-3.0 License https://ultralytics.com/license

import cv2

from utils.common import *
from utils.images import boundingRect


# @profile
def estimateAffine2D_SURF(im1, im2, p1, scale=1.0):
    """Estimates affine transformation between two images using SURF features; requires cv2, returns 2x3 matrix."""
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
        good = [m for m, n in matches if m.distance < 0.6 * n.distance]
        ngood = len(good)
        a += 10

    m1 = np.float32([kp1[x.queryIdx].pt for x in good]) + np.float32([x0, y0])
    m2 = np.float32([kp2[x.trainIdx].pt for x in good])
    # plots.imshow(im1 // 2 + im2 // 2, None, m1, m2)
    return cv2.estimateAffine2D(m1 / scale, m2 / scale, method=cv2.RANSAC)  # 2x3, better results


# @profile
def cv2calcOpticalFlowPyrLK(im1, im2, p1, p2hat=None, fbt=None, **lk_param):
    """Tracks keypoint motion between two images using Pyramidal Lucas-Kanade method; returns new points, status, and
    error.
    """
    # _, pyr1 = cv2.buildOpticalFlowPyramid(
    #    im0_roi, winSize=lk_param['winSize'], maxLevel=lk_param['maxLevel'], withDerivatives=True)
    # _, pyr2 = cv2.buildOpticalFlowPyramid(
    #    im_warped_0, winSize=lk_param['winSize'], maxLevel=lk_param['maxLevel'], withDerivatives=True)
    p2_, v, err = cv2.calcOpticalFlowPyrLK(im1, im2, p1, None, **lk_param)
    v = v.ravel().astype(bool)
    if fbt is not None:
        p1_, v2, _ = cv2.calcOpticalFlowPyrLK(im2, im1, p2_, None, **lk_param)
        fbe = norm(p1 - p1_, 1)
        v = v & v2.ravel().astype(bool) & (fbe < fbt)  # forward-backward error threshold
    return p2_, v, err


# @profile
def KLTregional(im0, im, p0, T, lk_param, fbt=1.0, translateFlag=False):
    """Tracks regional keypoints using the Kanade-Lucas-Tomasi (KLT) algorithm with forward-backward error
    thresholding.
    """
    T = T.astype(np.float32)
    # 1. Warp current image to past image frame
    x0, x1, y0, y1 = boundingRect(p0, im.shape, border=(50, 50))
    im0_roi = im0[y0:y1, x0:x1]
    xy0 = np.float32([x0, y0])
    p0_roi = p0 - xy0

    if translateFlag:
        dx = T[2, 0].__int__()
        dy = T[2, 1].__int__()
        im_warped_0 = im[y0 + dy : y1 + dy, x0 + dx : x1 + dx]
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
    pa, v, _ = cv2calcOpticalFlowPyrLK(im0_roi, im_warped_0, p0_roi, None, fbt=fbt, **lk_param)

    # convert p back to im coordinates
    if translateFlag:
        p = pa + (xy0 + [dx, dy]).astype(np.float32)
    else:
        p = addcol1(pa + xy0) @ T

    # residuals = norm(p0_roi[v] - pa[v],1)
    # _, i = fcnsigmarejection(residuals, srl=3, ni=3)
    # v[v] = i
    # plots.imshow(im_warped_0 // 2 + im0_roi // 2, None, p0_roi, pa)
    return p, v


# @profile
def KLTmain(im, im0, im0_small, p0):
    """Runs the Kanade-Lucas-Tomasi (KLT) feature tracking algorithm with coarse-to-fine tracking and affine
    transformation.
    """
    # Parameters for KLT
    EPS = cv2.TERM_CRITERIA_EPS
    COUNT = cv2.TERM_CRITERIA_COUNT
    lk_coarse = dict(winSize=(15, 15), maxLevel=4, criteria=(EPS | COUNT, 10, 0.1))
    lk_fine = dict(winSize=(51, 51), maxLevel=0, criteria=(EPS | COUNT, 30, 0.001))

    # 1. Coarse tracking on 1/8 scale full image
    scale = 1 / 4
    im_small = cv2.resize(im, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    if im0_small is None:
        im0_small = cv2.resize(im0, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    p, v, _ = cv2calcOpticalFlowPyrLK(im0_small, im_small, p0 * scale, None, **lk_coarse)
    p /= scale
    T23, inliers = cv2.estimateAffine2D(p0[v], p[v], method=cv2.RANSAC)  # 2x3, better results
    v[v] = inliers.ravel().astype(bool)
    # import plots; plots.imshow(im0_small//2+im_small//2, p1=p0[v]*scale,p2=p[v]*scale)

    # 2. Coarse tracking on full resolution roi https://www.mathworks.com/discovery/affine-transformation.html
    translation = p[v] - p0[v]
    T = np.eye(3, 2)
    T[2] = translation.mean(0)  # translation-only transform
    p, v = KLTregional(im0, im, p0, T, lk_coarse, fbt=1, translateFlag=True)

    if v.sum() > 10:  # good fit
        T23, inliers = cv2.estimateAffine2D(p0[v], p[v], method=cv2.RANSAC)  # 2x3, better results
    else:
        print("KLT coarse-affine failure, running SURF matches full scale.")
        T23, inliers = estimateAffine2D_SURF(im0, im, p0, scale=1)

    # 3. Fine tracking on affine-transformed regions
    p, v = KLTregional(im0, im, p0, T23.T, lk_fine, fbt=0.3)
    return p[v], v, im_small
