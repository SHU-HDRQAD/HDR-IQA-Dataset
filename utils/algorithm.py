import torch
import torch.nn.functional as F
import numpy as np
import os, cv2
from skimage.feature import greycomatrix, greycoprops
from math import exp
import random


PU_PLUT_FILENAME = './utils/data/PU_P_lut.npy'
PU_LLUT_FILENAME = './utils/data/PU_l_lut.npy'
COMP_MAXVAL = 1e4
COMP_MINVAL = 1e-6
crf_txt_file = './utils/data/dorfCurves.txt'


class pu_luts:
    def __init__(self):
        self.init_flag = False

    def init(self, p_lut_loc=None, l_lut_loc=None):
        if p_lut_loc is None:
            p_lut_loc = PU_PLUT_FILENAME
        if l_lut_loc is None:
            l_lut_loc = PU_LLUT_FILENAME
        self.P_lut = np.load(p_lut_loc)
        self.l_lut = np.load(l_lut_loc)
        self.init_flag = True

    def get_P_lut(self):
        if not self.init_flag:
            self.init()
        return self.P_lut

    def get_l_lut(self):
        if not self.init_flag:
            self.init()
        return self.l_lut


PU_lut = pu_luts()
SUGGEST_DORF_SEGMENTS ={
                        # 'A': (0, 52),
                        'B': (52, 72),
                        # 'C': (72, 112),
                        'D': (112, 142),
                        # 'E': (142, 152),
                        'F': (152, 172),
                        'G': (172, 200)
}  # Type 1: A, C, E with bigger contrast, Type 2: B, D, F, G with lower contrast


class CRFIdxGenerator:
    def __init__(self):
        self.ent = list(SUGGEST_DORF_SEGMENTS.keys())
        random.shuffle(self.ent)
        self.idx = -1

    def get(self):
        self.idx += 1
        if self.idx > (len(SUGGEST_DORF_SEGMENTS) - 1):
            self.idx = 0
            random.shuffle(self.ent)
        rang = SUGGEST_DORF_SEGMENTS[self.ent[self.idx]]
        return np.random.randint(rang[0], rang[1])


crf_idx_generator = CRFIdxGenerator()


class DoRF_CRF:
    def __init__(self, dorf_crf_file):
        self.init_flag = False
        self.crf_file = dorf_crf_file

    def parse_dorf(self, dorf_txt_loc):
        with open(os.path.join(dorf_txt_loc), 'r') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]

        i = [lines[idx + 3] for idx in range(0, len(lines), 6)]
        b = [lines[idx + 5] for idx in range(0, len(lines), 6)]
        type = [lines[idx] for idx in range(0, len(lines), 6)]
        info = [lines[idx + 1] for idx in range(0, len(lines), 6)]

        i = [ele.split() for ele in i]
        b = [ele.split() for ele in b]

        i = np.float32(i)
        b = np.float32(b)
        self.init_flag = True

        return i, b, type, info

    def random_crf(self, norm_inp, verbose=True):
        if not self.init_flag:
            self.ind, self.crf, self.type, _ = self.parse_dorf(self.crf_file)
        choice = np.random.randint(0, self.crf.shape[0])
        if verbose:
            print('Chosen CRF curve: %s.' % self.type[choice])
        return np.interp(norm_inp, self.ind[choice, :], self.crf[choice, :])

    def do_crf(self, norm_inp, id, verbose=False):
        if not self.init_flag:
            self.ind, self.crf, self.type, _ = self.parse_dorf(self.crf_file)
        if id >= 201:
            raise ValueError('Only 201 CRFs included in DoRF database, but given id = %d.' % id)
        choice = id
        if verbose:
            print('Chosen CRF curve: %s.' % self.type[choice])
        return np.interp(norm_inp, self.ind[choice, :], self.crf[choice, :])

    def get_crf_curve(self, idx):
        if not self.init_flag:
            self.ind, self.crf, self.type, _ = self.parse_dorf(self.crf_file)
        if idx >= 201:
            raise ValueError('Only 201 CRFs included in DoRF database, but given id = %d.' % id)
        return self.type[idx], self.ind[idx, :], self.crf[idx, :]


camera_transfer = DoRF_CRF(crf_txt_file)


def mu_law_inverse(x, mu=1200.0):
    mu = float(mu)
    if 'Tensor' in str(type(x)):
        return torch.log(1 + mu * x) / torch.tensor(np.log(1 + mu))
    else:
        return np.log(1 + mu * x) / np.log(1 + mu)


def mu_law_forward(x, mu=1200.0):
    mu = float(mu)
    if 'Tensor' in str(type(x)):
        out = torch.exp(torch.tensor(np.log(1 + mu)) * x) - 1
    else:
        out = np.exp(np.log(1 + mu) * x) - 1
    return out / mu


def pu(x):
    l_min = -5.0
    l_max = 10.0
    l = np.log10(np.maximum(np.minimum(np.power(10, l_max), x), np.power(10, l_min)))
    pu_l = 31.9270
    pu_h = 149.9244
    return (np.interp(l, PU_lut.get_l_lut(), PU_lut.get_P_lut()) - pu_l) / (pu_h - pu_l)


def psnr(gt, pred, range_=255.0):
    if 'Tensor' in str(type(pred)):
        mse = torch.mean(torch.pow((gt - pred), 2)) + 1e-7
        return 20 * torch.log10(torch.Tensor((range_,)) / torch.sqrt(mse))
    else:
        mse = np.mean((gt - pred) ** 2) + 1e-7
        return 20 * np.log10((range_) / np.sqrt(mse))


def _ssim_filt(img1, img2, ksize=11, sigma=1.5, maxval=255.0, output_cs=False):
    range_ = maxval
    C1 = (0.01 * range_) ** 2
    C2 = (0.03 * range_) ** 2

    img1 = img1.astype(np.float)
    img2 = img2.astype(np.float)
    kernel = cv2.getGaussianKernel(ksize, sigma)
    window = np.outer(kernel, kernel.transpose())

    p = ksize // 2
    mu1 = cv2.filter2D(img1, -1, window)[p:-p, p:-p]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[p:-p, p:-p]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[p:-p, p:-p] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[p:-p, p:-p] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[p:-p, p:-p] - mu1_mu2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    if not output_cs:
        return ssim_map.mean()
    else:
        return ssim_map.mean(), cs


def _gaussian(window_size, sigma):
    gauss = torch.from_numpy(np.array([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]))
    return gauss / gauss.sum()


def _create_window(window_size, channel, sigma):
    _1D_window = _gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, maxval, size_average=True, output_cs=False):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * maxval) ** 2
    C2 = (0.03 * maxval) ** 2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        s = ssim_map.mean()
    else:
        s = ssim_map.mean(1).mean(1).mean(1)

    if output_cs:
        return s, cs
    else:
        return s


def ssim(pre, ref, ksize, sigma, maxval=1.0):
    if 'Tensor' in str(type(pre)):
        (_, channel, _, _) = pre.size()
        window_size = ksize
        window = _create_window(window_size, channel, sigma)
        if pre.is_cuda:
            window = window.cuda(pre.get_device())
        window = window.type_as(pre)
        return _ssim(pre, ref, window, window_size, channel, maxval, size_average=True, output_cs=False)
    else:
        return _ssim_filt(pre, ref, ksize, sigma, maxval)


# from: https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py
def mssim_torch(img1, img2, ksize, sigma, maxval=1.0, size_average=True):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    ssims = []
    mcs = []

    (_, channel, _, _) = img1.size()
    window_size = ksize
    window = _create_window(window_size, channel, sigma)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    for _ in range(levels):
        sim, cs = _ssim(img1, img2, window, window_size, channel, maxval, size_average, True)
        ssims.append(sim)
        mcs.append(cs)
        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    ssims = torch.stack(ssims)
    mcs = torch.stack(mcs)
    pow1 = mcs ** weights
    pow2 = ssims ** weights

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1]) * pow2[-1]
    return output


def pq_inverse(inp):
    # make sure the inp is normalized into 0 ~ 1
    # HDR to LDR
    m1 = 0.1593
    m2 = 78.8438
    c1 = 0.8359
    c2 = 18.8516
    c3 = 18.6875

    if 'Tensor' in str(type(inp)):
        out = torch.pow((c1 + c2 * (torch.pow(inp, m1))) / (1 + c3 * (torch.pow(inp, m1))), m2)
    else:
        out = np.power((c1 + c2 * (np.power(inp, m1))) / (1 + c3 * (np.power(inp, m1))), m2)
    return out


def pq_forward(inp):
    # make sure the inp is normalized into 0 ~ 1
    # LDR to HDR
    M1 = 0.1593
    M2 = 78.8438
    C1 = 0.8359
    C2 = 18.8516
    C3 = 18.6875

    if 'Tensor' in str(type(inp)):
        out = torch.pow(torch.clip(torch.pow(inp, 1 / M2) - C1, 0, None) / (C2 - C3 * torch.pow(inp, 1 / M2)),
                        1 / M1)
    else:
        out = np.power((np.maximum((np.power(inp, 1 / M2) - C1), 0)) / (C2 - C3 * np.power(inp, (1 / M2))), 1 / M1)
    return out


def hlg_inverse(inp):
    # make sure the inp is normalized into 0 ~ 1
    # HDR to LDR
    a = 0.17883277
    b = 1 - 4 * a
    c = 0.5 - a * np.log(4*a)
    inp = np.select([inp > 1/12, inp <= 1/12], [a*np.log(12*inp-b)+c, np.sqrt(np.abs(3*inp))])
    return inp


def hlg_forward(inp):
    # make sure the inp is normalized into 0 ~ 1
    # LDR to HDR
    a = 0.17883277
    b = 1 - 4 * a
    c = 0.5 - a * np.log(4 * a)
    inp = np.select([inp > 1 / 2, inp <= 1 / 2], [(np.exp((inp-c) / a) + b) / 12.0, np.power(inp, 2.0) / 3.0])
    return inp


def map_range(x, low=0.0, high=1.0):
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)


def gram_matrix(x):
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def mask_low(a, th):
    if 'Tensor' in str(type(a)):
        return 0.5 * (1 - torch.tanh(200.0 * (a - th)))
    else:
        return 0.5 * (1 - np.tanh(200.0 * (a - th)))


def mask_high(a, th):
    if 'Tensor' in str(type(a)):
        return 0.5 * (1 + torch.tanh(200.0 * (a - th)))
    else:
        return 0.5 * (1 + np.tanh(200.0 * (a - th)))


def filter_high(a, th):
    return a * mask_low(a, th)


def filter_low(a, th):
    return a * mask_high(a, th)


def get_luminance(rgbimg):
    mat = [0.212639, 0.715169, 0.072192]
    if 'Tensor' in str(type(rgbimg)):
        l = mat[0] * rgbimg[:, 0, :, :] + mat[1] * rgbimg[:, 1, :, :] + mat[2] * rgbimg[:, 2, :, :]
        return l
    else:
        l = mat[0] * rgbimg[:, :, 0] + mat[1] * rgbimg[:, :, 1] + mat[2] * rgbimg[:, :, 2]
        return np.expand_dims(l, axis=-1)


def get_saturation(rgbimg):
    if 'Tensor' in str(type(rgbimg)):
        maxv = torch.max(rgbimg, dim=1, keepdim=True).values
        minv = torch.min(rgbimg, dim=1, keepdim=True).values
    else:
        maxv = np.max(rgbimg, axis=-1, keepdims=True)
        minv = np.min(rgbimg, axis=-1, keepdims=True)
    s = (maxv - minv) / (maxv + 1e-6)
    return s


def get_gradient(img):
    if 'Tensor' in str(type(img)):
        pad_H = F.pad(img, (0, 0, 0, 1), mode='reflect')
        pad_W = F.pad(img, (0, 1, 0, 0), mode='reflect')
        tv_H = torch.abs(pad_H[:, :, :-1, :] - pad_H[:, :, 1:, :])
        tv_W = torch.abs(pad_W[:, :, :, :-1] - pad_W[:, :, :, 1:])
    else:
        pad_H = np.pad(img, [[0, 1], [0, 0], [0, 0]], mode='reflect')
        pad_W = np.pad(img, [[0, 0], [0, 1], [0, 0]], mode='reflect')
        tv_H = np.abs(pad_H[:-1, :, :] - pad_H[1:, :, :])
        tv_W = np.abs(pad_W[:, :-1, :] - pad_W[:, 1:, :])
    return (tv_H + tv_W) / 2.0


def get_evLevel(rgbimg, mu=0.5, sigma=0.2):
    dem = 2 * sigma * sigma
    if 'Tensor' in str(type(rgbimg)):
        mu = torch.Tensor((mu,))
        img = torch.max(rgbimg, dim=1, keepdim=True)[0]
        out = -1.0 * torch.square(img - mu) / dem
        return torch.exp(out)
    else:
        img = np.max(rgbimg, axis=-1, keepdims=True)
        out = -1.0 * np.square(img - mu) / dem
        return np.exp(out)


def sobel_edge_mask(inp):
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')  #
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = torch.autograd.Variable(torch.from_numpy(sobel_kernel))
    inp = torch.max(inp, dim=1, keepdim=True)[0]
    mask = F.conv2d(inp, weight)
    mask = torch.abs(mask)
    mask = mask_high(mask, 0.1)
    return mask


def bcp(img, k):
    img = torch.max(img, dim=1, keepdim=True)[0]
    return F.max_pool2d(img, k, k)


class StretchContrastLevel(object):
    def __init__(self, threshLow=1/255.0, threshHigh=245/255.0, blackLevel=1/255.0, saturateLevel=245/255.0):
        self.th_low = threshLow
        self.th_high = threshHigh
        self.low = blackLevel
        self.high = saturateLevel

    def process(self, img):
        if img.max() < self.th_high:
            th_max = self.high
        else:
            th_max = img.max()
        if img.min() > self.th_low:
            th_min = self.low
        else:
            th_min = img.min()
        img = map_range(img, th_min, th_max)
        return img

    def __call__(self, img):
        return self.process(img)


class CRFProcess(object):
    def __init__(self, stops=0.0, crf_idx=100, randomize=False):
        if randomize:
            crf_idx = np.random.randint(0, 201)
        self.stops = stops
        self.crf = crf_idx

    def process(self, img):
        img = img * (2 ** self.stops)
        img = np.clip(img, 0, 1)
        img = camera_transfer.do_crf(img, self.crf, verbose=False)
        return img

    def get_crf_curve(self):
        return camera_transfer.get_crf_curve(self.crf)

    def __call__(self, img):
        return self.process(img)


class HardClip(object):
    def __init__(self, low_clip=0.0, high_clip=1.0):
        self.low = low_clip
        self.high = high_clip

    def process(self, img):
        return np.clip(img, self.low, self.high)

    def __call__(self, img):
        return self.process(img)


class OffsetMeanLevel(object):
    def __init__(self, mid_level=0.2, randomize=False):
        if randomize:
            mid_level = np.random.uniform(0.15, 0.51)
        self.level = mid_level

    def process(self, img):
        mean = np.mean(img)
        scale = self.level / mean
        img = img * scale
        return img

    def __call__(self, img):
        return self.process(img)


class PercentileClip(object):
    def __init__(self, low_clip=-1.0, high_clip=98.0, normalize=True, randomize=False):
        if randomize:
            low_clip = np.random.uniform(0.1, 2.0)
            high_clip = np.random.uniform(97.0, 99.0)
        if low_clip < 0.0 or low_clip > 100.0:
            self.enable_low = False
        else:
            self.enable_low = True
        self.low = low_clip
        if high_clip < 0.0 or high_clip > 100.0:
            self.enable_high = False
        else:
            self.enable_high = True
        self.high = high_clip
        self.normalize = normalize
        if self.enable_low is False and self.enable_high is False:
            print('Invalid PercentileClip ranges: %.1f - %.1f' % (low_clip, high_clip))
            exit(-1)

    def process(self, img):
        if not self.enable_low:
            high = np.percentile(img, self.high)
            img = np.clip(img, None, high)
        elif not self.enable_high:
            low = np.percentile(img, self.low)
            img = np.percentile(img, low, None)
        else:
            low, high = np.percentile(img, (self.low, self.high))
            img = np.clip(img, low, high)
        if self.normalize:
            return map_range(img, 0.0, 1.0)
        else:
            return img

    def __call__(self, img):
        return self.process(img)


class GammaInverse(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def process(self, img):
        return img ** (1.0 / self.gamma)

    def __call__(self, img):
        return self.process(img)


class Quantize(object):
    def __init__(self, bits):
        self.bits = bits

    def process(self, img):
        return np.fix(img * (2 ** self.bits)) / (2 ** self.bits)

    def __call__(self, img):
        return self.process(img)


class EVOffset(object):
    def __init__(self, ev):
        self.ev = ev

    def process(self, img):
        return img * (2 ** (-1.0 * self.ev))

    def __call__(self, img):
        return self.process(img)


class HSVProcess(object):
    def __init__(self, h_func, s_func, v_func):
        self.h = h_func
        self.s = s_func
        self.v = v_func

    def process(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        if self.h is not None:
            hsv[:, :, 0] = self.h(hsv[:, :, 0])
        if self.s is not None:
            hsv[:, :, 1] = self.s(hsv[:, :, 1])
        if self.v is not None:
            hsv[:, :, 2] = self.v(hsv[:, :, 2])
        im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return im

    def __call__(self, img):
        return self.process(img)


class Transforms:
    def __init__(self, func_list):
        self.funcs = func_list

    def __call__(self, img):
        for func in self.funcs:
            img = func(img)
        return img


class BaseTMO(object):
    def __init__(self, randomize=True):
        self.op = None
        self.params = {}
        self.random = randomize
        self.name = 'BaseTMO'

    def _rand_init(self):
        pass

    def __call__(self, img):
        if self.random:
            self._rand_init()
        return self.op.process(img)


class Reinhard(BaseTMO):
    def __init__(
        self,
        intensity=-1.0,
        light_adapt=0.8,
        color_adapt=0.0,
        gamma=2.0,
        randomize=False,
    ):
        super(Reinhard, self).__init__(randomize=randomize)
        self.op = cv2.createTonemapReinhard(
            gamma=gamma,
            intensity=intensity,
            light_adapt=light_adapt,
            color_adapt=color_adapt,
        )
        self.params['gamma'] = gamma
        self.params['intensity'] = intensity
        self.params['light_adapt'] = light_adapt
        self.params['color_adapt'] = color_adapt
        self.name = 'Reinhard'

    def _rand_init(self):
        gamma = np.random.uniform(2.0, 2.2)
        intensity = np.random.uniform(-1.0, 1.0)
        light_adapt = np.random.uniform(0.8, 1.0)
        color_adapt = np.random.uniform(0.0, 0.2)
        self.params['gamma'] = gamma
        self.params['intensity'] = intensity
        self.params['light_adapt'] = light_adapt
        self.params['color_adapt'] = color_adapt
        self.op = cv2.createTonemapReinhard(
            gamma=gamma,
            intensity=intensity,
            light_adapt=light_adapt,
            color_adapt=color_adapt,
        )


class Mantiuk(BaseTMO):
    def __init__(self, saturation=1.0, scale=0.75, gamma=2.0, randomize=False):
        super(Mantiuk, self).__init__(randomize=randomize)
        self.op = cv2.createTonemapMantiuk(
            saturation=saturation, scale=scale, gamma=gamma
        )
        self.params['saturation'] = saturation
        self.params['scale'] = scale
        self.params['gamma'] = gamma
        self.name = 'Mantiuk'

    def _rand_init(self):
        gamma = np.random.uniform(2.0, 2.2)
        scale = np.random.uniform(0.65, 0.85)
        saturation = np.random.uniform(0.85, 1.0)
        self.op = cv2.createTonemapMantiuk(
            saturation=saturation, scale=scale, gamma=gamma
        )
        self.params['saturation'] = saturation
        self.params['scale'] = scale
        self.params['gamma'] = gamma


class Drago(BaseTMO):
    def __init__(self, saturation=1.0, bias=0.85, gamma=2.0, randomize=False):
        super(Drago, self).__init__(randomize=randomize)
        self.op = cv2.createTonemapDrago(
            saturation=saturation, bias=bias, gamma=gamma
        )
        self.params['saturation'] = saturation
        self.params['bias'] = bias
        self.params['gamma'] = gamma
        self.name = 'Drago'

    def _rand_init(self):
        gamma = np.random.uniform(2.0, 2.2)
        bias = np.random.uniform(0.7, 0.9)
        saturation = np.random.uniform(0.85, 1.0)
        self.op = cv2.createTonemapDrago(
            saturation=saturation, bias=bias, gamma=gamma
        )
        self.params['saturation'] = saturation
        self.params['bias'] = bias
        self.params['gamma'] = gamma


class CRFToneMap(BaseTMO):
    def __init__(self, crf_idx=100, stops=0.0, gamma=2.2, randomize=False):
        super(CRFToneMap, self).__init__(randomize=randomize)
        if randomize:
            crf_idx = np.random.randint(0, 201)
            #gamma = np.random.uniform(2.05, 2.35)
        self.stops = stops
        self.crf = crf_idx
        self.gamma = gamma
        name, ind, val = self.get_crf_curve()
        self.params['EV'] = stops
        self.params['gamma'] = gamma
        self.params['crf'] = name
        self.params['crf_ind'] = ind
        self.params['crf_val'] = val
        self.op = self
        self.name = 'CRF'

    def process(self, img):
        img = img * (2 ** self.stops)
        img = np.clip(img, 0, 1)
        img = camera_transfer.do_crf(img, self.crf, verbose=False)
        return img ** (1.0 / self.gamma)

    def _rand_init(self):
        self.crf = np.random.randint(0, 201)
        self.stops = np.random.uniform(-0.5, 0.5)
        self.gamma = np.random.uniform(1.8, 2.2)
        name, ind, val = self.get_crf_curve()
        self.params['EV'] = self.stops
        self.params['gamma'] = self.gamma
        self.params['crf'] = name
        self.params['crf_ind'] = ind
        self.params['crf_val'] = val

    def get_crf_curve(self):
        return camera_transfer.get_crf_curve(self.crf)


TMO_DICT = {
    'crf': CRFToneMap,
    'reinhard': Reinhard,
    'mantiuk': Mantiuk,
    'drago': Drago,
}


def tonemap(img, tmo_name, **kwargs):
    out, _ = TMO_DICT[tmo_name](**kwargs)(img)
    return np.fix(map_range(out) * 255.0) / 255.0


def random_tonemap(x):
    tmos = list(TMO_DICT.keys())
    choice = np.random.randint(0, len(tmos))
    tmo = TMO_DICT[tmos[choice]](randomize=True)
    out = tmo(x)
    out = map_range(out)
    return np.fix(out * 255.0) / 255.0


class Masia17rTMO:
    def __init__(self, gamma=2.2, flag=True):
        self.gamma = gamma
        self.eps = 1e-6
        self.flag = flag

    def __call__(self, normalized_img):
        img = normalized_img
        img = img.astype(np.float32)
        img = img ** self.gamma
        L = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
        L_H = np.exp(np.log(L + self.eps).mean())
        oL1 = self._rid_outliner(L, 1)
        k1 = (np.log(L_H) - np.log(oL1.min())) / (np.log(oL1.max()) - np.log(oL1.min()))
        p_ov = np.prod(L[L > 254.0 / 255.0].shape) / np.prod(L.shape)
        gamma = 2.4379 + 0.2319 * np.log(L_H) - 1.1228 * k1 + 0.0085 * p_ov
        if self.flag:
            return img ** gamma
        else:
            return img ** (1.0 / gamma)

    def _rid_outliner(self, img, p):
        th_min, th_max = np.percentile(img, (p, 100 - p))
        im = img.copy()
        im[im < th_min] = th_min
        im[im > th_max] = th_max
        return im


def get_patch_complexity(pt):
    pt = np.max(pt, axis=-1)
    gre = greycomatrix(pt.astype(np.uint8), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    level = greycoprops(gre, 'contrast')
    return level[0, 0]


class HDRPU21:
    """PU-21 HDR encoding/decoding class

    Parameters
    ----------
    type: str, optional
        description string for determining the inner transfer model.
        Options are: 'banding', 'banding_glare' (default), 'peaks' and 'peaks_glare'.
    """
    def __init__(self, type='banding_glare'):
        self.L_min = 0.005  # The minimum linear value(luminance or radiance)
        self.L_max = 10000  # The maximum linear value(luminance or radiance)
        if type == 'banding':
            self.par = [1.070275272, 0.4088273932, 0.153224308, 0.2520326168, 1.063512885, 1.14115047, 521.4527484]
        elif type == 'banding_glare':
            self.par = [0.353487901, 0.3734658629, 8.277049286e-05, 0.9062562627, 0.09150303166, 0.9099517204,
                        596.3148142]
        elif type == 'peaks':
            self.par = [1.043882782, 0.6459495343, 0.3194584211, 0.374025247, 1.114783422, 1.095360363, 384.9217577]
        else:  ## type == 'peaks_glare
            self.par = [816.885024, 1479.463946, 0.001253215609, 0.9329636822, 0.06746643971, 1.573435413, 419.6006374]
        self.metric_type = type
        self.inverse = self.encode
        self.forward = self.decode

    def encode(self, x):
        """ PU encoding for the HDR image in absolute photometric units (nits).

            HDR images are often given in relative photometric units. They MUST be
            mapped to absolute amount of light emitted from the display. PU-21 requires
            the input HDR image to have the reference peak value as 10000.

        Parameters
        ----------
        x: 3-D ndarray
            the HDR image in absolute luminance whose reference peak value is 10000.

        Returns
        -------
        V: 3-D ndarry
            the encoded PU map (values range from 0 to 1)
        """
        x = np.minimum(np.maximum(x, self.L_min), self.L_max)
        p = self.par
        V = p[6] * (np.power((p[0] + p[1] * np.power(x, p[3])) / (1 + p[2] * np.power(x, p[3])), p[4]) - p[5])
        return V

    def decode(self, x):
        x = np.clip(x, 0, 1)
        p = self.par
        V_p = np.power(np.maximum(x / p[6] + p[5], 0), 1 / p[4])
        return np.power(np.maximum(V_p - p[0], 0) / (p[1] - p[2] * V_p), 1 / p[3])


class SDRPU21:
    """PU-21 SDR Encoding/Decoding class

    Parameters
    ----------
    Y_peak: int, optional
        display peak luminance in cd/m^2 (nit), e.g. 200 for a typical office monitor.
    contrast: int, optional
        the contrast of the display. The value 1000 means 1000:1.
    gamma: float, optional
        gamma of the display.
    E_ambient: int, optional
        ambient light illuminance in lux, e.g. 600 for bright office.
    k_refl: float, optional
        reflectivity of the display screen. Default is 0.005.
    pu_type: str, optional
        description string for determining the inner transfer model.
    """
    def __init__(self, Y_peak=100, contrast=1000, gamma=2.2, E_ambient=10, k_refl=0.005, pu_type='banding_glare'):
        self.Y_peak = Y_peak
        self.contrast = contrast
        self.gamma = gamma
        self.E_ambient = E_ambient
        self.k_refl = k_refl
        self.enc = HDRPU21(pu_type)
        self.inverse = self.encode
        self.forward = self.decode

    def get_black_level(self):
        Y_refl = self.E_ambient / np.pi * self.k_refl # Reflected ambient light
        Y_black = Y_refl + self.Y_peak / self.contrast
        return Y_black

    def linearize(self, V):
        Y_black = self.get_black_level()
        L = (self.Y_peak - Y_black) * np.power(V, self.gamma) + Y_black
        return L

    def encode(self, x):
        if x.max() > 1:
            x = x.astype(np.float32) / 255.0
        x = np.clip(x, 0, 1)
        x = self.linearize(x)
        return self.enc.inverse(x)

    def delinearize(self, V):
        Y_black = self.get_black_level()
        L = np.power((V - Y_black) / (self.Y_peak - Y_black), 1.0 / self.gamma)
        return L

    def decode(self, x, max_val=255.0):
        x = self.enc.forward(x)
        x = self.delinearize(x)
        return x * max_val
