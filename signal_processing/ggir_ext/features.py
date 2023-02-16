import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import math
import numpy as np
import pandas as pd

from .utils import get_ENMO, get_tilt_angles, get_LIDS
from .utils import mad, compute_entropy, get_diff_feat, get_stats, get_stats_current_window

def prep_data_and_compute_features(x, fs, time_interval=30):

  timestamps = np.arange(start=0, stop=x.shape[0] / fs, step=1/fs, dtype=float)
  data = np.concatenate((np.expand_dims(timestamps, axis=-1), x), axis=1)
  features = compute_features(data=data, time_interval=time_interval)
  return features


def compute_features(data, time_interval):
  df = pd.DataFrame(data, columns=['timestamp','x','y','z'])
  df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

  x = np.array(df['x'])
  y = np.array(df['y'])
  z = np.array(df['z'])
  timestamp = pd.Series(df['timestamp'])

  # Perform flipping x and y axes to ensure standard orientation
  # For correct orientation, x-angle should be mostly negative
  # So, if median x-angle is positive, flip both x and y axes
  # Ref: https://github.com/wadpac/hsmm4acc/blob/524743744068e83f468a4e217dde745048a625fd/UKMovementSensing/prepacc.py
  angx = np.arctan2(x, np.sqrt(y*y + z*z)) * 180.0/math.pi
  if np.median(angx) > 0:
      x *= -1
      y *= -1

  ENMO = get_ENMO(x,y,z)
  angle_x, angle_y, angle_z = get_tilt_angles(x,y,z)
  LIDS = get_LIDS(timestamp, ENMO)

  _, ENMO_stats = get_stats(df['timestamp'], ENMO, time_interval)
  _, angle_z_stats = get_stats(df['timestamp'], angle_z, time_interval)
  timestamp_agg, LIDS_stats = get_stats(df['timestamp'], LIDS, time_interval)
  feat = np.hstack((ENMO_stats, angle_z_stats, LIDS_stats))

  return feat

def prep_data_and_compute_surrogate_signals(x, fs, time_interval=30):

  timestamps = np.arange(start=0, stop=x.shape[0] / fs, step=1/fs, dtype=float)
  data = np.concatenate((np.expand_dims(timestamps, axis=-1), x), axis=1)
  features = compute_surrogate_signals(data=data, time_interval=time_interval)
  return features

def compute_surrogate_signals(data, time_interval):
  df = pd.DataFrame(data, columns=['timestamp','x','y','z'])
  df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

  x = np.array(df['x'])
  y = np.array(df['y'])
  z = np.array(df['z'])
  timestamp = pd.Series(df['timestamp'])

  # Perform flipping x and y axes to ensure standard orientation
  # For correct orientation, x-angle should be mostly negative
  # So, if median x-angle is positive, flip both x and y axes
  # Ref: https://github.com/wadpac/hsmm4acc/blob/524743744068e83f468a4e217dde745048a625fd/UKMovementSensing/prepacc.py
  angx = np.arctan2(x, np.sqrt(y*y + z*z)) * 180.0/math.pi
  if np.median(angx) > 0:
      x *= -1
      y *= -1

  ENMO = get_ENMO(x,y,z)
  angle_x, angle_y, angle_z = get_tilt_angles(x,y,z)
  LIDS = get_LIDS(timestamp, ENMO)

  #_, ENMO_stats = get_stats_current_window(df['timestamp'], ENMO, time_interval)
  #_, angle_z_stats = get_stats_current_window(df['timestamp'], angle_z, time_interval)
  #timestamp_agg, LIDS_stats = get_stats_current_window(df['timestamp'], LIDS, time_interval)
  #feat = np.hstack((ENMO_stats, angle_z_stats, LIDS_stats))

  surrogate_signals = np.vstack((ENMO, angle_z, LIDS)).swapaxes(1, 0)

  return surrogate_signals