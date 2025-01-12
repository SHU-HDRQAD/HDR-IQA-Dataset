import utils.exrio as eio
import cv2
import skimage.io
import numpy as np
import rawpy
import exifread
import pylab as pl
import matplotlib.pyplot as plt
import re


EV_REQ_FLAGS = [
    'EXIF ExposureTime',
    'EXIF FNumber',
    'EXIF ISOSpeedRatings',
    'EXIF ExposureBiasValue',
    'EXIF MaxApertureValue',
    'EXIF FocalLength'
]


def read_ev_bias_from_img_exif(img_loc, tag_id='EXIF ExposureBiasValue'):
    with open(img_loc, 'rb') as f:
        tag = exifread.process_file(f, stop_tag=tag_id, details=False)
    target = tag[tag_id].values[0]
    return float(target.numerator) / float(target.denominator)


def read_et_frac_from_img_exif(img_loc, tags='EXIF ExposureTime'):
    with open(img_loc, 'rb') as f:
        exif = exifread.process_file(f, stop_tag=tags, details=False)
    return exif[tags].values[0].denominator


def read_ev_info_from_exif(img_loc, tags=None):
    if tags is None:
        tags = EV_REQ_FLAGS
    with open(img_loc, 'rb') as f:
        exif = exifread.process_file(f, stop_tag=tags[-1], details=False)
    et = exif['EXIF ExposureTime'].values[0].den
    fnum = exif['EXIF FNumber'].values[0]
    fnum = float(fnum.numerator) / float(fnum.denominator)
    iso = exif['EXIF ISOSpeedRatings'].values[0]
    ebv = exif['EXIF ExposureBiasValue'].values[0]
    ebv = float(ebv.numerator) / float(ebv.denominator)
    map = exif['EXIF MaxApertureValue'].values[0]
    map = float(map.numerator) / float(map.denominator)
    fl = exif['EXIF FocalLength'].values[0]
    fl = float(fl.numerator) / float(fl.denominator)
    return et, fnum, iso, ebv, map, fl


def load(img_dir):
    hdr_suffix = ['pfm', 'exr', 'hdr', 'dng', 'PFM', 'EXR', 'HDR', 'DNG']
    ldr_suffix = ['jpg', 'png', 'jpeg', 'bmp', 'JPG', 'PNG', 'JPEG', 'BMP']
    if any([img_dir[-3:] == k for k in hdr_suffix]):
        img = load_HDR(img_dir)
        img = img.astype(np.float32)
    elif any([img_dir[-3:] == k for k in ldr_suffix]):
        img = load_LDR(img_dir)
        img = img.astype(np.float32) / 255.0
    else:
        with open(img_dir, 'r') as f:
            img = f.read()
    return img


def load_HDR(hdr_dir):
    if 'dng' in hdr_dir:
        rio = rawpy.imread(hdr_dir)
        pixels = rio.postprocess(use_camera_wb=True,
                                 use_auto_wb=False,
                                 no_auto_bright=True,
                                 output_bps=16).astype(np.float32)
        rio.close()
    elif 'exr' in hdr_dir:
        pixels = eio.read(hdr_dir)
    else:
        pixels = cv2.imread(hdr_dir, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_ANYCOLOR)
        if pixels is None:
            print("Read_image error: ", hdr_dir)
            exit(-1)
        pixels = np.flip(pixels, 2)
    return pixels


def load_LDR(ldr_dir):
    img = cv2.imread(ldr_dir)
    if img is None:
        print("Read_image error:", ldr_dir)
        exit(-1)
    img = np.flip(img, 2)
    return img


def save_HDR(hdr_dir, pixels):
    if 'tiff' in hdr_dir:
        pixels = np.flip(pixels, 2).astype(np.uint16)
        ret = cv2.imwrite(hdr_dir, pixels)
        if ret is False:
            print('Error occurred in saving image: ' + hdr_dir)
            exit(1)
    elif 'exr' in hdr_dir:
        eio.save(hdr_dir, pixels)
    else:
        pixels = np.flip(pixels, 2).astype(np.float32)
        ret = cv2.imwrite(hdr_dir, pixels)
        if ret is False:
            print('Error occurred in saving image: ' + hdr_dir)
            exit(1)


def save_LDR(ldr_dir, pixels):
    skimage.io.imsave(ldr_dir, pixels.astype(np.uint8))


def save_normalized_ldr(loc, img, gamma=2.2):
    inpt_ldr = np.squeeze(np.clip(np.power(img, 1 / gamma), 0, 1)) * 255
    save_LDR(loc, inpt_ldr)


def save_normalized_hdr(loc, img):
    inpt_ldr = np.squeeze(np.clip(img, 0, 1))
    if 'tiff' in loc:
        inpt_ldr = inpt_ldr * 65535.0
    save_HDR(loc, inpt_ldr)


def image_resize(image, resize):
    height, width, channel = image.shape
    if channel > 3:
        image = image[:, :, :3]
    exp_h = resize[0]
    exp_w = resize[1]
    im2 = cv2.resize(image, (exp_h, exp_w), interpolation=cv2.INTER_LINEAR)
    im2[im2 < 0] = 0
    return im2


# copied from: https://www.jianshu.com/p/9194f43fd68a
def cv_show(name, img):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey()
    # cv2.destroyAllWindows()


# copied from: https://www.jianshu.com/p/9194f43fd68a
def image_rotate(image, angle):
    (h, w) = image.shape[:2]
    (cx, cy) = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy

    return cv2.warpAffine(image, M, (nW, nH))


def map_range(x, low=0, high=1, dtype=np.float32):
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(dtype)


def replace_specials_(x, val=0.0005):
    x[np.isinf(x).sum() | np.isnan(x).sum()] = val
    return x


def add_noise(x, noise_factor=0.1):
    return x + noise_factor * x.std() * np.random.random(x.shape)


def save_normalized_images(save_path, stacked_imgs, type='ldr', gamma=2.2, stack_style='horizontal',
                           transfer_func=None):
    if transfer_func is not None:
        imgs = transfer_func(stacked_imgs)
    else:
        imgs = stacked_imgs
    imgs = np.squeeze(imgs)
    if type == 'ldr':
        imgs = np.clip(np.power(imgs, 1 / gamma), 0, 1) * 255
        save_handle = save_LDR
    else:
        save_handle = save_HDR
    if len(imgs.shape) == 3:
        save_handle(save_path, imgs)
    else:
        st = imgs.shape[0]
        h = imgs.shape[1]
        w = imgs.shape[2]
        if stack_style == 'horizontal':
            tmp = np.zeros([h * st, w, 3], dtype=np.float32)
            for i in range(st):
                tmp[i * h: (i + 1) * h, :, :] = imgs[i, :, :, :]
        else:
            tmp = np.zeros([h, w * st, 3], dtype=np.float32)
            for i in range(st):
                tmp[:, i * w: (i + 1) * w, :] = imgs[i, :, :, :]
        save_handle(save_path, tmp)


def draw_hist(image, minval, maxval, bin_num, savename, name='image', xlabel='Luminance Range',
              ylabel='Histogram Occurences'):
    lenths = np.reshape(image, (-1, 1))
    data = lenths
    bins = np.linspace(minval, maxval, bin_num)
    pl.hist(data, bins)
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    pl.title(name)
    # pl.show()
    pl.savefig(savename)
    pl.close()


def show_distribution(inp, bins, minval=None, maxval=None, name_axis_x='Range', name_axis_y='Occurance',
                      title='Value Distribution'):
    if maxval is None:
        maxval = inp.max()
    if minval is None:
        minval = inp.min()
    maxvals_ = np.reshape(inp, (-1, 1))
    bins_ = np.linspace(minval, maxval, bins)
    pl.hist(maxvals_, bins_)
    pl.xlabel(name_axis_x)
    pl.ylabel(name_axis_y)
    pl.title(title)
    pl.show()


def plot_function(x, y, name_axis_x='input', name_axis_y='Output', title='Function', color='b', marker='.'):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title(title)
    plt.xlabel(name_axis_x)
    plt.ylabel(name_axis_y)
    ax1.scatter(x.flatten(), y.flatten(), c=color, marker=marker)
    plt.show()


def flip_ud(img):
    return cv2.flip(img, 1)


def flip_lr(img):
    return cv2.flip(img, 0)
