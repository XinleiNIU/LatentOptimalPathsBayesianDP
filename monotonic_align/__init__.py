import numpy as np
import torch
from .monotonic_align.core import maximum_path_c
import pdb

def maximum_path(pi, t_x_len, t_y_len):  
  """ Cython optimised version.
  value: [b, t_x, t_y]
  mask: [b, t_x, t_y]
  """
  device = pi.device
  dtype = pi.dtype
  pi = pi.data.cpu().numpy().astype(np.float32)
  path = np.zeros((pi.shape[0],pi.shape[1],pi.shape[2])).astype(np.int32)

  t_x_len = t_x_len.data.cpu().numpy().astype(np.int32)
  t_y_len = t_y_len.data.cpu().numpy().astype(np.int32)
  maximum_path_c(path, pi, t_x_len, t_y_len)
  return torch.from_numpy(path).to(device=device, dtype=dtype)
