__author__ = 'v-penlu'
import numpy as np
import math
from numpy.linalg import norm
import SequenceToSequence as STS

# text vocab range(10)

# in_vocab_size = 4
# out_vocab_szie = 5
# hidden_size_list = [30, 20]
# we_size = 2
# params = STS.Construct_net(hidden_size_list, we_size, in_vocab_size,
#                            out_vocab_szie)
# sample = [[0, 3, 3, 2, 1], [0, 4, 1, 2], [4, 1, 2, 0]]
# STS.Grad_check(params, sample)

vocab_size = 5
hidden_size_list = [30, 30]
we_size = 5
params = STS.Construct_LM_net(hidden_size_list, we_size, vocab_size, embedding_range=1)
sample = [[0, 3, 4, 1, 2], [3, 4, 1, 2, 0]]
STS.Grad_check_LM(params, sample)

# check softmax layer
# sigFigs = 6
# time_steps = 30
# lower_layer_size = 5
# layer_size = 3
# joint_size = lower_layer_size + 1
# rng = np.random.RandomState(89757)
# lower_output_acts = range(time_steps)
# for i in range(time_steps):
#     lower_output_acts[i] = np.asarray(rng.uniform(low=-1, high=1,
#                                                   size=(lower_layer_size, 1)))
# W_o = np.asarray(rng.uniform(low=-0.1, high=0.1,
#                              size=(layer_size, joint_size)))
# target_idx_seq = np.asarray(rng.randint(layer_size, size=time_steps)).tolist()
# softmax_ret = STS.Softmax_feed_fordward_backward(W_o, lower_output_acts,
#                                                  target_idx_seq)
# algDg = softmax_ret[0]
#
# pert = 1e-5
# diff1 = np.zeros(W_o.shape)
# diff2 = np.zeros(W_o.shape)
# for i in range(W_o.shape[0]):
#     for j in range(W_o.shape[1]):
#         W_o[i, j] += pert
#         _, _, diff1[i, j] = STS.Softmax_feed_fordward_backward(W_o,
#                                 lower_output_acts, target_idx_seq)
#         W_o[i, j] -= 2 * pert
#         _, _, diff2[i, j] = STS.Softmax_feed_fordward_backward(W_o,
#                                 lower_output_acts, target_idx_seq)
#         W_o[i, j] += pert
# numDg = (diff1 - diff2) / (2 * pert)
#
# threshold = np.power(10.0, np.maximum(0.0, np.ceil(np.log10(np.minimum(np.abs(algDg), np.abs(numDg)))))-int(sigFigs))
# diff = np.abs(algDg - numDg)
# np.savetxt("W_o.diff.txt", diff)
# np.savetxt("W_o.D.txt", algDg)
# np.savetxt("W_o.numD.txt", numDg)
# np.savetxt("threshold.txt", threshold)
# if np.sum(np.abs(diff) > threshold) > 0:
#     print "softmax check failed"
