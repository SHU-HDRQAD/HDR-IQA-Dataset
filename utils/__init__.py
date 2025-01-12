import utils.metrics as mts
import torch
import time
import os
import importlib
import utils.algorithm as alg
import utils.image_io as iio
import utils.misc as misc
from utils.misc import LossesHelper, cv2torch, torch2cv, torch_eval


LOGGER_POOL = {'default': misc.Recorder('default')}


def get_records(file, notify):
    rcd = []
    try:
        with open(file, 'r') as f:
            rcd = f.readlines()
            rcd = [n.strip() for n in rcd]
    except OSError:
        if notify:
            print('Warning: record file %s is not available.' % file)
    return rcd


def probe_gpu(options, notify=True):
    gpu = False
    multi_card = False
    if torch.cuda.is_available() and 'gpu_ids' in options.keys():
        gpu = True
        if len(options['gpu_ids']) > 1:
            multi_card = True
    if gpu:
        if notify:
            print('Device: GPU %s' % options['gpu_ids'])
        return torch.device('cuda:' + options['gpu_ids']), gpu, multi_card
    else:
        if notify:
            print('Device: CPU')
        return torch.device('cpu'), gpu, multi_card


def get_metric_handle(opt):
    metrics = opt['metrics']
    if isinstance(metrics, str):
        metrics = [metrics]
    m_list = {}
    for m in metrics:
        if m == 'psnr' or m == 'PSNR':
            m_list['PSNR'] = mts.PSNR()
        elif m == 'ssim' or m == 'SSIM':
            m_list['SSIM'] = mts.SSIM()
        elif m == 'pq-psnr' or m == 'PQ-PSNR':
            m_list['PQ-PSNR'] = mts.PQPSNR()
        elif m == 'pq-ssim' or m == 'PQ-SSIM':
            m_list['PQ-SSIM'] = mts.PQSSIM()
        elif m == 'vdp2' or m == 'VDP2':
            path = opt['vdp2_matlab_path']
            m_list['HDR-VDP2'] = mts.VDP2(matlab_path=path)
        elif m == 'PSNR-MU' or m == 'psnr-mu':
            mu = int(opt['metric_mu_val'])
            m_list['PSNR-MU%d' % mu] = mts.PSNR_MU(mu)
        elif m == 'SSIM-MU' or m == 'ssim-mu':
            mu = int(opt['metric_mu_val'])
            m_list['SSIM-MU%d' % mu] = mts.SSIM_MU(mu)
        elif m == 'd-psnr' or m == 'D-PSNR':
            m_list['D-PSNR'] = mts.DiffMetric(mts.PSNR())
        elif m == 'd-ssim' or m == 'D-SSIM':
            m_list['D-SSIM'] = mts.DiffMetric(mts.SSIM())
        elif m == 'd-pqpsnr' or m == 'D-PQPSNR':
            m_list['D-PQPSNR'] = mts.DiffMetric(mts.PQPSNR())
        elif m == 'd-pqssim' or m == 'D-PQSSIM':
            m_list['D-PQSSIM'] = mts.DiffMetric(mts.PQSSIM())
        elif m == 'd-vdp2' or m == 'D-VDP2':
            path = opt['vdp2_matlab_path']
            m_list['D-HDR-VDP2'] = mts.DiffMetric(mts.VDP2(matlab_path=path))
        elif m == 'D-PSNR-MU' or m == 'd-psnr-mu':
            mu = int(opt['metric_mu_val'])
            m_list['D-PSNR-MU%d' % mu] = mts.DiffMetric(mts.PSNR_MU(mu))
        elif m == 'D-SSIM-MU' or m == 'd-ssim-mu':
            mu = int(opt['metric_mu_val'])
            m_list['D-SSIM-MU%d' % mu] = mts.DiffMetric(mts.SSIM_MU(mu))
        else:
            raise RuntimeError('Not supported metric: %s.' % m)
    return m_list


class MetricEvaluation:
    def __init__(self, opt):
        self.option = opt
        id = 'metrics'
        if id not in opt.keys() or opt[id] == 'none' or opt[id] is None or len(opt[id]) == 0:
            self.enabled = False
            self.m_list = {}
            self.logger = None
            self.counts = 0
            print('No metric is designated for evaluation.')
        else:
            self.enabled = True
            self.m_list = get_metric_handle(opt)
            self.logger = LossesHelper(list(self.m_list.keys()))
            self.counts = len(self.m_list.keys())

    def process(self, label, gt, pred):
        if self.enabled:
            outs = []
            for key in self.m_list.keys():
                outs.append(self.m_list[key](label, pred, gt))
            return self.logger.iter_record(outs, False)
        else:
            return ''

    def flush(self):
        if self.enabled:
            self.logger.flush()

    def report_all_averages(self, verbose=True):
        if self.enabled:
            return self.logger.report_all_averages(verbose)
        else:
            return ''

    def report_all_std(self, verbose=True):
        if self.enabled:
            return self.logger.report_all_std(verbose)
        else:
            return ''


def get_logfile(opt, stage):
    if stage == 'train':
        model_loc = opt['checkpoint_dir']
    else:
        model_loc = opt['save_location']
    model_name = opt['model_name']
    return os.path.join(model_loc, time.strftime(model_name + '_' + stage + '_%y-%m-%d_%H:%M:%S.txt'))


def get_procedure_handle(opt):
    proc_handle = opt['procedure_handle']
    params = importlib.import_module(proc_handle)
    if hasattr(params, 'train_process'):
        t_handle = getattr(params, 'train_process')
    else:
        t_handle = None
        RuntimeError('No "%s.%s" handle existed, please specify the train procedure.' % (proc_handle,
                                                                                         'train_process'))
    if hasattr(params, 'eval_process'):
        e_handle = getattr(params, 'eval_process')
    else:
        e_handle = None
        RuntimeError('No "%s.%s" handle existed, please specify the train procedure.' % (proc_handle,
                                                                                         'eval_process'))
    return t_handle, e_handle


def get_save_handle(opt):
    save_handle = opt['save_handle']
    params = importlib.import_module(save_handle)
    if hasattr(params, 'result'):
        result = getattr(params, 'result')
    else:
        result = None
        print('No "%s.%s" save handle existed, samples saving skipped.' % (save_handle, 'result'))
    return result


def get_logger(name):
    if name in LOGGER_POOL.keys():
        return LOGGER_POOL[name]
    else:
        t = misc.Recorder(name)
        LOGGER_POOL[name] = t
        return t
