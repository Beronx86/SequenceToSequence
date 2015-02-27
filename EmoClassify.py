__author__ = 'Liu'
import itertools
import numpy as np
real = np.float64

def Cos_feed_forward(params, in_seq_1, in_seq_2, tag = 0, pool_len=0, average=False):
    """Maximize cosine distance between different emotion class and minimize cosine
    distance between same emotion class.
    :param params:
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
        num_frames_1 = [len(in_seq_1) / pool_len] * pool_len
        num_frames_1[-1] += len(in_seq_1) % pool_len
        frame_idx_1 = itertools.accumulate(num_frames_1)
        frame_idx_1.insert(0, 0)
        num_frames_2 = [len(in_seq_2) / pool_len] * pool_len
        num_frames_2[-1] += len(in_seq_2) % pool_len
        frame_idx_2 = itertools.accumulate(num_frames_2)
        frame_idx_2.insert(0, 0)
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
        
