# general imports
import math
import numpy as np
from typing import *

# tf imports
import tensorflow as tf

def unadjusted_mse(y_true, y_pred):
    y_pred = tf.reshape(y_pred, y_true.shape)
    interim = tf.math.square(y_pred - y_true).astype('float64')
    return tf.math.reduce_mean(interim)

def unadjusted_mae(y_true, y_pred):
    y_pred = tf.reshape(y_pred, y_true.shape)
    interim = tf.math.abs(y_pred - y_true).astype('float64')
    return tf.math.reduce_mean(interim)

def distance_adjusted_mse(inputs, y_true, y_pred, training_radius: float, alpha: float = 1.0):
    y_pred = tf.reshape(y_pred, y_true.shape)

    d = tf.math.abs(inputs - training_radius) / training_radius
    d = tf.math.maximum(d, 1.0)
    distance_coeff = tf.math.pow(tf.math.reciprocal(d), alpha).astype('float64')
    interim = tf.math.multiply(tf.math.square(y_pred - y_true).astype('float64'), distance_coeff)
    return tf.math.reduce_mean(interim)

def distance_adjusted_mae(inputs, y_true, y_pred, training_radius: float, alpha: float = 1.0):
    y_pred = tf.reshape(y_pred, y_true.shape)

    d = tf.math.abs(inputs - training_radius) / training_radius
    d = tf.math.maximum(d, 1.0)
    distance_coeff = tf.math.pow(tf.math.reciprocal(d), alpha).astype('float64')
    interim = tf.math.multiply(tf.math.abs(y_pred - y_true).astype('float64'), distance_coeff)
    return tf.math.reduce_mean(interim)

def least_metrics_by_periodic_shift(epsilon: float, variant, inputs, predictions, training_radius: float, metrics: List[Callable], y_normalizer: float, sample_count: int = 100):
    shifts = np.linspace(-epsilon, +epsilon, 2*sample_count+1, endpoint=True)
    leasts = list(+math.inf for metric in metrics)
    for shift in shifts:
        shift_counts = np.floor_divide(inputs, training_radius)
        particular_shifts = shift * shift_counts
        shifted_inputs = inputs + particular_shifts

        y_true = variant.execute(x=shifted_inputs) / y_normalizer
        for i, metric in zip(range(0, len(metrics)), metrics):
            value = metric(inputs=inputs, y_true=y_true, y_pred=predictions, training_radius=training_radius)
            leasts[i] = min(value, leasts[i])
    
    return leasts

def least_metrics_by_periodic_speedup(epsilon: float, variant, inputs, predictions, training_radius: float, metrics: List[Callable], y_normalizer: float, sample_count: int = 100):
    speedups = np.linspace(-epsilon, +epsilon, 2*sample_count+1, endpoint=True)
    leasts = list(+math.inf for metric in metrics)
    for speedup in speedups:
        particular_speedups = (1 - speedup)
        spedup_inputs = inputs * particular_speedups

        y_true = variant.execute(x=spedup_inputs) / y_normalizer
        for i, metric in zip(range(0, len(metrics)), metrics):
            value = metric(inputs=inputs, y_true=y_true, y_pred=predictions, training_radius=training_radius)
            leasts[i] = min(value, leasts[i])
    
    return leasts

def least_metrics_by_periodic_acceleration(epsilon: float, variant, inputs, predictions, training_radius: float, metrics: List[Callable], y_normalizer: float, sample_count: int = 100):
    speedups = np.linspace(-epsilon, +epsilon, 2*sample_count+1, endpoint=True)
    leasts = list(+math.inf for metric in metrics)
    for speedup in speedups:
        speedup_counts = np.floor_divide(inputs, training_radius)
        particular_speedups = (1 - speedup) * speedup_counts
        spedup_inputs = inputs * particular_speedups

        y_true = variant.execute(x=spedup_inputs) / y_normalizer
        for i, metric in zip(range(0, len(metrics)), metrics):
            value = metric(inputs=inputs, y_true=y_true, y_pred=predictions, training_radius=training_radius)
            leasts[i] = min(value, leasts[i])
    
    return leasts