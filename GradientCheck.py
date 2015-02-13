__author__ = 'v-penlu'
import numpy as np
import math
from numpy.linalg import norm


import SequenceToSequence as STS

# text vocab range(10)
in_vocab_size = 10
out_vocab_szie = 14
hidden_size_list = [11, 12]
we_size = 9
params = STS.Construct_net(hidden_size_list, we_size, in_vocab_size,
                           out_vocab_szie)
sample = [[7, 6, 5, 4, 3, 2, 1], [0, 8, 9, 2, 3], [8, 9, 2, 3, 0]]
STS.Grad_check(params, sample)

# check softmax layer
"""
time_steps = 30
lower_layer_size = 5
layer_size = 3
joint_size = lower_layer_size + 1
rng = np.random.RandomState(89757)
lower_output_acts = range(time_steps)
for i in range(time_steps):
    lower_output_acts[i] = np.asarray(rng.uniform(low=-1, high=1,
                                                  size=(lower_layer_size, 1)))
W_o = np.asarray(rng.uniform(low=-0.1, high=0.1,
                             size=(layer_size, joint_size)))
target_idx_seq = np.asarray(rng.randint(layer_size, size=time_steps)).tolist()
softmax_ret = STS.Softmax_feed_fordward_backward(W_o, lower_output_acts,
                                                 target_idx_seq)
Dg_o = softmax_ret[0]

mu = 2 * 1e-9 * (1 + norm(W_o, 2))
diff1 = np.zeros(W_o.shape)
diff2 = np.zeros(W_o.shape)
for i in range(W_o.shape[0]):
    for j in range(W_o.shape[1]):
        W_o[i, j] += mu
        _, _, diff1[i, j] = STS.Softmax_feed_fordward_backward(W_o,
                                lower_output_acts, target_idx_seq)
        W_o[i, j] -= 2 * mu
        _, _, diff2[i, j] = STS.Softmax_feed_fordward_backward(W_o,
                                lower_output_acts, target_idx_seq)
        W_o[i, j] += mu
numDg_o = (diff1 - diff2) / (2 * mu)

diff = np.abs(Dg_o - numDg_o)
np.savetxt("W_o.diff.txt", diff)
np.savetxt("W_o.D.txt", Dg_o)
np.savetxt("W_o.numD.txt", numDg_o)
"""
