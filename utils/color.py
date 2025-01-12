# color transfer
import numpy as np
import cv2


RGB709_XYZ = np.array([
    [0.412391, 0.357584, 0.180481],
    [0.212639, 0.715169, 0.072192],
    [0.019331, 0.119195, 0.950532]
    ])


RGB2020_XYZ = np.array([
    [0.636958, 0.144617, 0.168881],
    [0.262700, 0.677998, 0.059302],
    [0.000000, 0.028073, 1.060985]
    ])


RGBP3D65_XYZ = np.array([
    [0.486571, 0.265668, 0.198217],
    [0.228975, 0.691739, 0.079287],
    [0.000000, 0.045113, 1.043944]
    ])


XYZ_RGB709 = np.array([
    [3.240970, -1.537383,  -0.498611],
    [-0.969244,  1.875968,   0.041555],
    [0.055630, -0.203977,   1.056972]
    ])


XYZ_RGB2020 = np.array([
    [1.716651, -0.355671, -0.253366],
    [-0.666684,  1.616481,  0.015768],
    [0.017640, -0.042771,  0.942103]
    ])


XYZ_RGBP3D65 = np.array([
    [2.493497, -0.931384, -0.402711],
    [-0.829489, 1.762664, 0.023625],
    [0.035846, 0.076172, 0.956885]
    ])


XYZ_LMS = np.array([[0.359283259012122, 0.697605114777950, -0.035891593232029],
                    [-0.192080846370499, 1.100476797037432, 0.075374865851912],
                    [0.007079784460748, 0.074839666218637, 0.843326545389877]
                    ])


LMS_XYZ = np.array([[2.070152218389423, -1.326347338967156, 0.206651047629405],
                    [0.364738520974807, 0.680566024947228, -0.045304545922035],
                    [-0.049747207535812, -0.049260966696614, 1.188065924992303]
                    ])


LMS_IPT = np.array([[0.5, 0.5, 0],
                    [1.613769531, -3.323486328, 1.709716797],
                    [4.378173828, -4.245605469, -0.132568359]
                    ])


IPT_LMS = np.array([[1.0, 0.008609037037933, 0.111029625003026],
                    [1.0, -0.008609037037933, -0.111029625003026],
                    [1.0, 0.560031335710679, -0.320627174987319]
                    ])


# Suppose rgbimg has size [H, W, 3] with permutation of [r, g, b] in the 3th axis
def extract_luminance(rgbimg, colorspace=0):
    method = False
    if(method):
        l = rgbimg.max(axis=2)
    else:
        if colorspace == 1:
            mat = RGB2020_XYZ[1]
        elif colorspace == 2:
            mat = RGBP3D65_XYZ[1]
        else:
            mat = RGB709_XYZ[1]
        l = mat[0] * rgbimg[:, :, 0] + mat[1] * rgbimg[:, :, 1] + mat[2] * rgbimg[:, :, 2]
    return l


def divide_luma_chroma(img, colorspace=0):
    luma = extract_luminance(img, colorspace=colorspace)
    chroma = img / np.stack([luma, luma, luma], axis=2)
    return luma, chroma


def merge_luma_chroma(luma, chroma, gamma=1.0, saturation=1.0):
    result = np.stack([luma, luma, luma], axis=2) * chroma
    return result


def luma_operation(rgbimg, op, saturation=1.0):
    hsv = cv2.cvtColor(rgbimg, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, -1]
    s = hsv[:, :, -2]
    v_ = op(v)
    hsv_ = np.stack([hsv[:, :, 0], saturation * s, v_], axis=-1)
    rgb_ = cv2.cvtColor(hsv_, cv2.COLOR_HSV2RGB)
    return rgb_


def colorspace_transfer(img, trmat):
    c0 = trmat[0][0] * img[:, :, 0] + trmat[0][1] * img[:, :, 1] + trmat[0][2] * img[:, :, 2]
    c1 = trmat[1][0] * img[:, :, 0] + trmat[1][1] * img[:, :, 1] + trmat[1][2] * img[:, :, 2]
    c2 = trmat[2][0] * img[:, :, 0] + trmat[2][1] * img[:, :, 1] + trmat[2][2] * img[:, :, 2]
    return np.stack([c0, c1, c2], axis=2)


# From GitHub (bilateral grid)
# From: https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
RGB_TO_YUV = np.array([
    [ 0.299,     0.587,     0.114],
    [-0.168736, -0.331264,  0.5],
    [ 0.5,      -0.418688, -0.081312]])
YUV_TO_RGB = np.array([
    [1.0,  0.0,      1.402],
    [1.0, -0.34414, -0.71414],
    [1.0,  1.772,    0.0]])
YUV_OFFSET = np.array([0, 0.5, 0.5]).reshape(1, 1, -1)


def rgb2yuv(im, maxval):
    return np.tensordot(im.astype(float), RGB_TO_YUV, ([2], [1])) + YUV_OFFSET * maxval


def yuv2rgb(im, maxval):
    return np.tensordot(im.astype(float) - YUV_OFFSET * maxval, YUV_TO_RGB, ([2], [1]))
