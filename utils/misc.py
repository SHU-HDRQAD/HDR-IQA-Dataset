import os, re
import numpy as np
import torch
import torch.nn.functional as F
import time
import utils.image_io as iio
from utils.algorithm import map_range


# All from github repo: hdr-expandnet
def cv2torch(np_img):
    return torch.from_numpy(np_img.swapaxes(1, 2).swapaxes(0, 1))


def torch2cv(t_img):
    t_img = t_img.cpu()
    return t_img.numpy().swapaxes(0, 2).swapaxes(0, 1)


def process_path(directory, create=False):
    directory = os.path.expanduser(directory)
    directory = os.path.normpath(directory)
    directory = os.path.abspath(directory)
    if create:
        try:
            os.makedirs(directory)
        except:
            pass
    return directory


def split_path(directory):
    directory = process_path(directory)
    name, ext = os.path.splitext(os.path.basename(directory))
    return os.path.dirname(directory), name, ext


# From torchnet
# https://github.com/pytorch/tnt/blob/master/torchnet/transform.py
def compose(transforms):
    'Composes list of transforms (each accept and return one item)'
    assert isinstance(transforms, list)
    for transform in transforms:
        assert callable(transform), 'list of functions expected'

    def composition(obj):
        'Composite function'
        for transform in transforms:
            obj = transform(obj)
        return obj

    return composition


def check_valid_input_size(factor_of_k, h, w):
    if h % factor_of_k != 0 or w % factor_of_k != 0:
        return False
    return True


def resize_image_to_factor_of_k(img, k=64, mode='bilinear'):
    b, c, h, w = img.shape
    new_h = int(np.ceil(h / float(k)) * k)
    new_w = int(np.ceil(w / float(k)) * k)
    img = F.interpolate(input=img, size=(new_h, new_w), mode=mode, align_corners=False)
    return img


def torch_eval(file_dir, tensor):
    max_ = tensor.max().item()
    min_ = tensor.min().item()
    if len(tensor.shape) == 4:
        tensor = tensor[0, ...]
    t = torch2cv(tensor)
    iio.save_LDR(file_dir, 255 * map_range(t[:, :, :3]))
    return min_, max_


def set_random_seed(seed):
    torch.manual_seed(seed)


def save_to_file(filename, a):
    with open(filename, 'a') as f:
        f.writelines(a)
        f.write('\n')


def list_file_pairs(ldr_dir, hdr_dir, suffix1, suffix2):
    # ldr_suffix = ".*.(jpg|png|jpeg|bmp)"
    # hdr_suffix = ".*.(pfm|exr|hdr|dng)"
    t1 = os.listdir(ldr_dir)
    t2 = os.listdir(hdr_dir)
    reg_ldr = re.compile(suffix1)
    reg_hdr = re.compile(suffix2)
    ldr_list = [f.split('.') for f in t1 if reg_ldr.match(f)]
    hdr_list = [f.split('.') for f in t2 if reg_hdr.match(f)]
    l_list = []
    h_list = []
    ldr_list.sort(key=(lambda x: x[0]))
    hdr_list.sort(key=(lambda x: x[0]))
    if len(ldr_list) != len(hdr_list):
        print("Error: dataset pairs count uneven.")
        exit(1)
    for i in range(len(ldr_list)):
        if ldr_list[i][0] != hdr_list[i][0]:
            print("Error: dataset picture names unpair.")
            exit(1)
        l_list.append(os.path.join(ldr_dir, ldr_list[i][0] + '.' + ldr_list[i][1]))
        h_list.append(os.path.join(hdr_dir, hdr_list[i][0] + '.' + hdr_list[i][1]))
    return l_list, h_list, len(l_list)


class LossesHelper:
    def __init__(self, loss_dec):
        self.count = len(loss_dec)
        self.loss_pool = []
        for i in range(self.count):
            self.loss_pool.append([])
        self.decs = loss_dec

    def iter_record(self, entries, display=True):
        if len(entries) != self.count:
            print('Not valid losses entry!')
            exit(-1)
        rcd_str = ''
        for i in range(self.count):
            self.loss_pool[i].append(entries[i])
            rcd_str += '%s %.5f' % (self.decs[i], entries[i])
            if i < (self.count - 1):
                rcd_str += ', '
            else:
                rcd_str += '.'
        if display:
            print(rcd_str)
        return rcd_str

    def last_record(self, display=True):
        rcd_str = ''
        for i in range(self.count):
            rcd_str += '%s %.5f' % (self.decs[i], self.loss_pool[i][-1])
            if i < (self.count - 1):
                rcd_str += ', '
            else:
                rcd_str += '.'
        if display:
            print(rcd_str)
        return rcd_str

    def get_average(self, i):
        if i >= self.count or i < 0:
            print('Not valid index number!')
            exit(-1)
        m_result = np.mean(np.array(self.loss_pool[i]))
        return m_result

    def get_std(self, i):
        if i >= self.count or i < 0:
            print('Not valid index number!')
            exit(-1)
        m_result = np.std(np.array(self.loss_pool[i]))
        return m_result

    def get_maximum(self, i):
        if i >= self.count or i < 0:
            print('Not valid index number!')
            exit(-1)
        return max(self.loss_pool[i])

    def get_minimum(self, i):
        if i >= self.count or i < 0:
            print('Not valid index number!')
            exit(-1)
        return min(self.loss_pool[i])

    def report_average(self, i, display=True):
        res = self.get_average(i)
        rcd_str = '%s %.5f' % (self.decs[i], res)
        if display:
            print(rcd_str)
        return rcd_str

    def report_std(self, i, display=True):
        res = self.get_std(i)
        rcd_str = '%s %.5f' % (self.decs[i], res)
        if display:
            print(rcd_str)
        return rcd_str

    def report_maximum(self, i, display=True):
        res = self.get_maximum(i)
        rcd_str = '%s %.5f' % (self.decs[i], res)
        if display:
            print(rcd_str)
        return rcd_str

    def report_minimum(self, i, display=True):
        res = self.get_minimum(i)
        rcd_str = '%s %.5f' % (self.decs[i], res)
        if display:
            print(rcd_str)
        return rcd_str

    def flush(self):
        for i in range(self.count):
            self.loss_pool[i].clear()

    def report_all_averages(self, display=True):
        rcd_str = ''
        for i in range(self.count):
            rcd_str += self.report_average(i, False)
            if i < (self.count - 1):
                rcd_str += ', '
            else:
                rcd_str += '.'
        if display:
            print(rcd_str)
        return rcd_str

    def report_all_std(self, display=True):
        rcd_str = ''
        for i in range(self.count):
            rcd_str += self.report_std(i, False)
            if i < (self.count - 1):
                rcd_str += ', '
            else:
                rcd_str += '.'
        if display:
            print(rcd_str)
        return rcd_str

    def report_all_maximum(self, display=True):
        rcd_str = ''
        for i in range(self.count):
            rcd_str += self.report_maximum(i, False)
            if i < (self.count - 1):
                rcd_str += ', '
            else:
                rcd_str += '.'
        if display:
            print(rcd_str)
        return rcd_str

    def report_all_minimum(self, display=True):
        rcd_str = ''
        for i in range(self.count):
            rcd_str += self.report_minimum(i, False)
            if i < (self.count - 1):
                rcd_str += ', '
            else:
                rcd_str += '.'
        if display:
            print(rcd_str)
        return rcd_str


class Timer:
    def __init__(self, name='Execution'):
        self._time = 0
        self._perf = LossesHelper(['%s time' % name])
        self.clear = True
        self.name = name

    def tic(self):
        self._time = time.time()

    def toc(self, display=True):
        self.clear = False
        t = time.time() - self._time
        if self._time == 0:
            if display:
                print('Warning: %s timer is not initialized by tic().' % self.name)
        self._perf.iter_record([t], display)
        return t

    def average(self, display=True):
        if not self.clear:
            return self._perf.report_all_averages(display)
        else:
            return 'Time: 0.0000.'

    def reset(self, display=True):
        rcd = self.average(display)
        self.clear = True
        self._time = 0
        self._perf.flush()
        return rcd


class ImagePad:
    def __init__(self, base_length):
        self.base_length = base_length
        self.pad_flag = False
        self.pad_l = 0
        self.pad_r = 0
        self.pad_d = 0
        self.pad_u = 0

    def pad(self, img):
        h = img.shape[-2]
        w = img.shape[-1]
        if h % self.base_length != 0:
            self.pad_u = (self.base_length - (h % self.base_length)) // 2
            self.pad_d = self.base_length - (h % self.base_length) - self.pad_u
            self.pad_flag = True
        if w % self.base_length != 0:
            self.pad_l = (self.base_length - (w % self.base_length)) // 2
            self.pad_r = self.base_length - (w % self.base_length) - self.pad_l
            self.pad_flag = True
        if self.pad_flag:
            img = F.pad(img, (self.pad_l, self.pad_r, self.pad_u, self.pad_d), mode='constant', value=0)
        return img

    def depad(self, img):
        h = img.shape[-2]
        w = img.shape[-1]
        return img[..., self.pad_u:(h - self.pad_d), self.pad_l:(w - self.pad_r)]


class Recorder:
    def __init__(self, name='My Recoder', file_loc='./data'):
        self._name = name
        self._loc = file_loc

    def set_loc(self, loc):
        self._loc = loc
        return self

    def __call__(self, title, notes, display=True):
        if title == '':
            save_to_file(self._loc, notes)
            if display:
                print(notes)
        else:
            save_to_file(self._loc, title + ': ' + notes)
            if display:
                print(title + ': ')
                print(notes)
