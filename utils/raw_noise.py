import numpy as np
import torch


NOISE_PARAMS = {
    'SIDD': {
        'log_min_shot_noise': 0.00068674,
        'log_max_shot_noise': 0.02194856,
        'read_variance': 0.20,
        'composite_linear_k': 1.85,
        'composite_linear_b': 0.30
    },
    'DND': {
        'log_min_shot_noise': 0.0001,
        'log_max_shot_noise': 0.012,
        'read_variance': 0.26,
        'composite_linear_k': 2.18,
        'composite_linear_b': 1.20
    }
}


def random_normal(mean, std, r):
    if 'Tensor' in str(type(r)):
        distribution = torch.distributions.normal.Normal(mean, std)
        return distribution.sample()
    else:
        return np.random.normal(mean, std)


def random_uniform(low, high, r):
    if 'Tensor' in str(type(r)):
        distribution = torch.distributions.uniform.Uniform(low, high)
        return distribution.sample()
    else:
        return np.random.uniform(low, high)


def _pow(k, x, r):
    if 'Tensor' in str(type(r)):
        return torch.pow(k, x)
    else:
        return np.power(k, x)


def _log10(x, r):
    if 'Tensor' in str(type(r)):
        return torch.log10(x)
    else:
        return np.log10(x)


def _sqrt(x, r):
    if 'Tensor' in str(type(r)):
        return torch.sqrt(x)
    else:
        return np.sqrt(x)


def raw_noise_params(scheme):
    if scheme == 'DND':
        params = NOISE_PARAMS['DND']
    elif scheme == 'SIDD':
        params = NOISE_PARAMS['SIDD']
    else:
        params = {
            'log_min_shot_noise': np.random.uniform(0.0001, 0.001),
            'log_max_shot_noise': np.random.uniform(0.012, 0.022),
            'read_variance': np.random.uniform(0.20, 0.28),
            'composite_linear_k': np.random.uniform(1.80, 2.20),
            'composite_linear_b': np.random.uniform(0.30, 1.20)
        }
    return params


def generate_raw_noise(raw, params):
    sh_min = params['log_min_shot_noise']
    sh_max = params['log_max_shot_noise']
    rv = params['read_variance']
    k = params['composite_linear_k']
    b = params['composite_linear_b']
    log_min_shot_noise = _log10(sh_min, raw)
    log_max_shot_noise = _log10(sh_max, raw)
    log_shot_noise = random_uniform(log_min_shot_noise, log_max_shot_noise, raw)
    shot_noise = _pow(10, log_shot_noise, raw)
    read_noise = random_normal(0.0, rv, raw)
    line = lambda x: k * x + b
    log_read_noise = line(log_shot_noise) + read_noise
    read_noise = _pow(10, log_read_noise, raw)
    return shot_noise, read_noise


def add_noise(image, shot_noise, read_noise):
  """Adds random shot (proportional to image) and read (independent) noise."""
  variance = image * shot_noise + read_noise
  mean = 0.0
  noise = random_normal(mean, _sqrt(variance, image), image)
  return image + noise


def raw_random_noise(image, scheme):
    params = raw_noise_params(scheme)
    shot, read = generate_raw_noise(image, params)
    return add_noise(image, shot, read)
