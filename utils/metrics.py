import utils.algorithm as alg
import transplant
import numpy as np


class PSNR:
    def __init__(self, maxval=1.0):
        self.max = maxval

    def __call__(self, i, x, y):
        return alg.psnr(x, y, self.max)


class SSIM:
    def __init__(self, maxval=1.0):
        self.max = maxval

    def __call__(self, i, x, y):
        return alg.ssim(x, y, 11, 1.5, self.max)


class PQPSNR(PSNR):
    def __call__(self, i, x, y):
        x = alg.pq_inverse(x / self.max)
        y = alg.pq_inverse(y / self.max)
        return alg.psnr(x, y, 1.0)


class PQSSIM(SSIM):
    def __call__(self, i, x, y):
        x = alg.pq_inverse(x / self.max)
        y = alg.pq_inverse(y / self.max)
        return alg.ssim(x, y, 11, 1.5, 1.0)


class PSNR_MU:
    def __init__(self, mu=5000.0, maxval=1.0):
        self.max = maxval
        self.mu = mu

    def __call__(self, i, x, y):
        x = alg.mu_law_inverse(x / self.max, self.mu)
        y = alg.mu_law_inverse(y / self.max, self.mu)
        return alg.psnr(x, y, 1.0)


class SSIM_MU:
    def __init__(self, mu=5000.0, maxval=1.0):
        self.max = maxval
        self.mu = mu

    def __call__(self, i, x, y):
        x = alg.mu_law_inverse(x / self.max, self.mu)
        y = alg.mu_law_inverse(y / self.max, self.mu)
        return alg.ssim(x, y, 11, 1.5, 1.0)


class VDP2:
    def __init__(self, maxval=1.0, norm=False, matlab_path='../hdrvdp-2.2.1'):
        self.matlab = transplant.Matlab(jvm=False, desktop=False)
        self.matlab.addpath(matlab_path)
        self.per_norm = norm
        self.max = maxval

    def __call__(self, i, inp, lbl):
        if self.per_norm:
            inp = inp / inp.max()
            lbl = lbl / lbl.max()
        else:
            inp = inp / self.max
            lbl = lbl / self.max
        return self.matlab.metric_vdp2(inp.astype(np.float32), lbl.astype(np.float32))


class DiffMetric:
    def __init__(self, met_handle):
        self.handle = met_handle

    def __call__(self, i, x, y):
        return self.handle(None, x, y) - self.handle(None, i, y)
