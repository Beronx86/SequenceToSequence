__author__ = 'Liu'
import itertools
import numpy as np
import SequenceToSequence as STS
from scipy.spatial.distance import cosine
import math
real = np.float64


def Calculate_frame_idx(time_steps, pool_len):
    num_frames = [time_steps / pool_len] * pool_len
    num_frames[-1] += time_steps % pool_len
    frame_idx = itertools.accumulate(num_frames)
    frame_idx.insert(0, 0)
    return frame_idx


def Pool_feed_forward(in_seq_1, in_seq_2, pool_len=0, average=False):
    """Maximize cosine distance between different emotion class and minimize cosine
    distance between same emotion class.
    :param in_seq_1:
    :param in_seq_2:
    :param tag: 1 is positive sample, 0 is negative sample
    :param pool_len:
    :param average: if pool_len=0 and average=True, average the sum
                    if pool_len=0 and average=False, use the sum directly
                    if pool_len=k and average=True, average pooling
                    if pool_len=k and average=False, max pooling
    :return:
    """
    feature_vec_1 = 0
    feature_vec_2 = 0
    max_idx_1 = 0
    max_idx_2 = 0
    if pool_len == 0:
        feature_vec_1 = 0
        feature_vec_2 = 0
        for v in in_seq_1:
            feature_vec_1 += v
        for v in in_seq_2:
            feature_vec_2 += v
        if average:
            feature_vec_1 /= len(in_seq_1)
            feature_vec_2 /= len(in_seq_2)
    else:
        feature_dim = in_seq_2[0].shape[0]
        frame_idx_1 = Calculate_frame_idx(len(in_seq_1), pool_len)
        frame_idx_2 = Calculate_frame_idx(len(in_seq_2), pool_len)
        feature_mtx_1 = np.zeros((len(in_seq_1), feature_dim), dtype=real)
        feature_mtx_2 = np.zeros((len(in_seq_2), feature_dim), dtype=real)
        for i, v in enumerate(in_seq_1):
            feature_mtx_1[i] = in_seq_1[i][0]
        for i, v in enumerate(in_seq_2):
            feature_mtx_2[i] = in_seq_2[i][0]
        feature_vec_1 = np.zeros((feature_dim * pool_len), dtype=real)
        feature_vec_2 = np.zeros((feature_dim * pool_len), dtype=real)
        if average:
            for i in range(pool_len):
                s = feature_dim * i
                e = s + feature_dim
                feature_vec_1[s: e] = np.average(feature_mtx_1[frame_idx_1[i]: frame_idx_1[i + 1]], axis=0)
                feature_vec_2[s: e] = np.average(feature_mtx_2[frame_idx_2[i]: frame_idx_2[i + 1]], axis=0)
        else:
            for i in range(pool_len):
                s = feature_dim * i
                e = s + feature_dim
                feature_vec_1[s: e] = np.max(feature_mtx_1[frame_idx_1[i]: frame_idx_1[i + 1]], axis=0)
                feature_vec_2[s: e] = np.max(feature_mtx_2[frame_idx_2[i]: frame_idx_2[i + 1]], axis=0)
                # In max pooling back propagation, only errors only propagate to
                # max units.
                max_idx_1 = np.argmax(feature_mtx_1[frame_idx_1[i]: frame_idx_1[i + 1]], axis=0)
                max_idx_2 = np.argmax(feature_mtx_2[frame_idx_2[i]: frame_idx_2[i + 1]], axis=0)
    return feature_vec_1, feature_vec_2, max_idx_1, max_idx_2


def Pool_feed_backward(input_errs_1, input_errs_2, time_steps_1, time_steps_2,
                       max_idx_1=0, max_idx_2=0, pool_len=0, average=False):
    Dl_pool_1 = range(time_steps_1)
    Dl_pool_2 = range(time_steps_2)
    if pool_len == 0:
        feature_dim = input_errs_1.shape[0]
        if not average:
            for i in range(time_steps_1):
                Dl_pool_1[i] = input_errs_1.reshape(feature_dim, 1)
            for i in range(time_steps_2):
                Dl_pool_2[i] = input_errs_2.reshape(feature_dim, 1)
        else:
            for i in range(time_steps_1):
                Dl_pool_1[i] = (input_errs_1 / time_steps_1).reshape(feature_dim, 1)
            for i in range(time_steps_2):
                Dl_pool_2[i] = (input_errs_2 / time_steps_2).reshape(feature_dim, 1)
    else:
        feature_dim = input_errs_1.shape[0] / pool_len
        frame_idx_1 = Calculate_frame_idx(time_steps_1, pool_len)
        frame_idx_2 = Calculate_frame_idx(time_steps_2, pool_len)
        if average:
            for i in range(pool_len):
                s = feature_dim * i
                e = s + feature_dim
                for j in range(frame_idx_1[i], frame_idx_1[i + 1]):
                    Dl_pool_1[j] = input_errs_1[s: e].reshape(feature_dim, 1)
                for j in range(frame_idx_2[i], frame_idx_2[i + 1]):
                    Dl_pool_2[j] = input_errs_2[s: e].reshape(feature_dim, 1)
        else:
            Dl_mtx_1 = np.zeros((time_steps_1, feature_dim))
            Dl_mtx_2 = np.zeros((time_steps_2, feature_dim))
            for i in range(pool_len):
                s = feature_dim * i
                e = s + feature_dim
                c_idx = range(feature_dim)
                r_idx = max_idx_1[i]
                Dl_mtx_1[r_idx, c_idx] = input_errs_1[s: e]
                r_idx = max_idx_2[i]
                Dl_mtx_2[r_idx, c_idx] = input_errs_2[s: e]
            for i in range(time_steps_1):
                Dl_pool_1[i] = Dl_mtx_1[i].reshape(feature_dim, 1)
            for i in range(time_steps_2):
                Dl_pool_2[i] = Dl_mtx_2[i].reshape(feature_dim, 1)
    return Dl_pool_1, Dl_pool_2


def Cos_feed_forward(vec_1, vec_2, is_posative):
    vec_1_pow = vec_1.dot(vec_1)
    vec_2_pow = vec_2.dot(vec_2)
    vec_1_norm = math.sqrt(vec_1_pow)
    vec_2_norm = math.sqrt(vec_2_pow)
    vec_1_2_p = vec_1.dot(vec_2)
    cos = vec_1_2_p / (vec_1_norm * vec_2_norm)
    if is_posative:
        loss = (1 - cos) / 2
    else:
        loss = cos * cos
    cos2 = 1 - cosine(vec_1, vec_2)
    Dl_1 = vec_2 / (vec_1_norm * vec_2_norm) - vec_1 * cos / vec_1_pow
    Dl_2 = vec_1 / (vec_1_norm * vec_2_norm) - vec_2 * cos / vec_2_pow
    return loss, Dl_1, Dl_2



def Bi_Collapse_forward(forward_acts, backward_acts):
    """ Combine 2 direction actions into 1 vector.
    :param forward_acts: form previous to future time_idx 1, 2, 3, ..., t
    :param backward_acts: form future to previous time_idx t, t-1, t-2, ..., 0
    :return:
    """
    time_stemps = len(forward_acts)
    half_layer_size = forward_acts[0].shape[0]
    out_acts = range(time_stemps)
    for i in range(time_stemps):
        out_acts[i] = np.zeros((half_layer_size * 2, 1), dtype=real)
        out_acts[i][0: half_layer_size] = forward_acts[i]
        out_acts[i][half_layer_size: 2 * half_layer_size] = backward_acts[time_stemps - 1 - i]
    return out_acts


def Bi_Split_backward(input_errs):
    """ Split the error to 2 directions
    :param input_errs:
    :return:
    """
    time_steps = len(input_errs)
    half_layer_size = input_errs[0].shape[0] / 2
    forward_input_errs = range(time_steps)
    backward_input_errs = range(time_steps)
    for i in range(time_steps):
        forward_input_errs[i] = input_errs[i][0: half_layer_size]
        backward_input_errs[time_steps - 1 - i] = input_errs[i][half_layer_size: 2 * half_layer_size]
    return forward_input_errs, backward_input_errs


def Bi_Collapse_backward(f_lower_input_errs, b_lower_input_errs):
    for f_errs, b_errs in zip(f_lower_input_errs, reversed(b_lower_input_errs)):
        f_errs += b_errs
    return f_errs


def Bi_LSTM_feed_forward(f_weights, b_weights, lower_output_acts):
    f_inter_vals = STS.LSTM_feed_forward(f_weights, lower_output_acts)
    b_inter_vals = STS.LSTM_feed_forward(b_weights, lower_output_acts[::-1])
    output_acts = Bi_Collapse_forward(f_inter_vals[0], b_inter_vals[0])
    return output_acts, f_inter_vals, b_inter_vals


def Bi_LSTM_feed_backward(f_weights, b_weights, f_inter_vals, b_inter_vals,
                          input_errors):
    f_input_errs, b_input_errs = Bi_Split_backward(input_errors)
    f_Dg, f_errs, _ = STS.LSTM_feed_backward(f_weights, f_inter_vals, f_input_errs)
    b_Dg, b_errs, _ = STS.LSTM_feed_backward(b_weights, b_inter_vals, b_input_errs)
    lower_input_errs = Bi_Collapse_backward(f_errs, b_errs)
    return f_Dg, b_Dg, lower_input_errs
