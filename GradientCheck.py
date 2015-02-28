__author__ = 'v-penlu'
import numpy as np
import math
from numpy.linalg import norm
import SequenceToSequence as STS
import LanguageModel as LM
real = np.float64

check_STS_2vocab = 0
check_STS_1vocab = 0
check_LM = 0
check_softmax = 0
check_Generage = 0
check_Beamsearch = 0
check_parallel = 1
rng = np.random.RandomState()

if check_parallel:
    in_vocab_size = 20
    out_vocab_size = 19
    hidden_size_list = [8, 9]
    we_size = 19
    sample1 = "2 9 8 10 15 6 1|3 7 16 3 4 9 1"
    sample2 = "19 3 7 1|17 13 1"
    lines = [sample1, sample2]
    in_batch, out_batch, target_batch, in_seq_lens, out_seq_lens = STS.Construct_batch(lines)
    params = STS.Construct_net(hidden_size_list, we_size, in_vocab_size, out_vocab_size)
    STS.Feed_forward_backward(params, in_batch, out_batch, target_batch,
                              in_seq_lens, out_seq_lens)

if check_STS_2vocab:
    em_time_steps = 11
    lm_time_steps = 7
    in_vocab_size = 27
    out_vocab_szie = 33
    hidden_size_list = [8, 9]
    we_size = 19
    params = STS.Construct_net(hidden_size_list, we_size, in_vocab_size,
                               out_vocab_szie, embedding_range=1)
    in_vocab = rng.randint(in_vocab_size, size=em_time_steps).tolist()
    out = rng.randint(low=1, high=out_vocab_szie, size=lm_time_steps).tolist()
    sample = [in_vocab, [0] + out, out + [0]]
    STS.Grad_check(params, sample)

if check_STS_1vocab:
    em_time_steps = 11
    lm_time_steps = 9
    vocab_size = 27
    hidden_size_list = [22, 17]
    we_size = 15
    params = STS.Construct_net(hidden_size_list, we_size, vocab_size,
                               embedding_range=1)
    in_vocab = rng.randint(vocab_size, size=em_time_steps).tolist()
    out = rng.randint(low=1, high=vocab_size, size=lm_time_steps).tolist()
    sample = [in_vocab, [0] + out, out + [0]]
    STS.Grad_check(params, sample)

if check_Generage:
    em_time_steps = 11
    lm_time_steps = 9
    vocab_size = 27
    hidden_size_list = [22, 17]
    we_size = 15
    params = STS.Construct_net(hidden_size_list, we_size, vocab_size,
                               embedding_range=1)
    in_vocab = rng.randint(vocab_size, size=em_time_steps).tolist()
    gen = STS.Generate(params, in_vocab)


if check_Beamsearch:
    em_time_steps = 11
    lm_time_steps = 9
    vocab_size = 27
    hidden_size_list = [22, 17]
    we_size = 15
    params = STS.Construct_net(hidden_size_list, we_size, vocab_size,
                               embedding_range=1)
    in_vocab = rng.randint(vocab_size, size=em_time_steps).tolist()
    gen = STS.Beam_search_generate(params, in_vocab)


# Check LM is mainly to check the lstm feed forward and backward
if check_LM:
    vocab_size = 80
    time_steps = 20
    hidden_size_list = [11, 21]
    we_size = 39
    rand_sample = rng.randint(low=1, high=vocab_size, size=time_steps)
    params = LM.Construct_LM_net(hidden_size_list, we_size, vocab_size,
                                 embedding_range=1, lstm_range=0.1)
    in_seq = [0] + rand_sample.tolist()
    out_seq = rand_sample.tolist() + [0]
    sample = [in_seq, out_seq]
    LM.Grad_check_LM(params, sample)

# check softmax layer
if check_softmax:
    sigFigs = 6
    time_steps = 20
    lower_layer_size = 5
    layer_size = 3
    joint_size = lower_layer_size + 1
    lower_output_acts = range(time_steps)
    for i in range(time_steps):
        lower_output_acts[i] = np.asarray(rng.uniform(low=-0.1, high=0.1,
                                                      size=(lower_layer_size, 1)),
                                          dtype=real)
    W_o = np.asarray(rng.uniform(low=-0.1, high=0.1,
                                 size=(layer_size, joint_size)), dtype=real)
    target_idx_seq = np.asarray(rng.randint(layer_size, size=time_steps)).tolist()
    softmax_ret = STS.Softmax_feed_fordward_backward(W_o, lower_output_acts,
                                                     target_idx_seq)
    algDg = softmax_ret[0]

    pert = 1e-6
    diff1 = np.zeros(W_o.shape)
    diff2 = np.zeros(W_o.shape)
    for i in range(W_o.shape[0]):
        for j in range(W_o.shape[1]):
            W_o[i, j] += pert
            _, _, diff1[i, j] = STS.Softmax_feed_fordward_backward(W_o,
                                    lower_output_acts, target_idx_seq)
            W_o[i, j] -= 2 * pert
            _, _, diff2[i, j] = STS.Softmax_feed_fordward_backward(W_o,
                                    lower_output_acts, target_idx_seq)
            W_o[i, j] += pert
    numDg = (diff1 - diff2) / (2 * pert)

    threshold = np.power(10.0, np.maximum(0.0, np.ceil(np.log10(np.minimum(np.abs(algDg), np.abs(numDg)))))-int(sigFigs))
    diff = np.abs(algDg - numDg)
    np.savetxt("W_o.diff.txt", diff)
    np.savetxt("W_o.D.txt", algDg)
    np.savetxt("W_o.numD.txt", numDg)
    np.savetxt("threshold.txt", threshold)
    if np.sum(np.abs(diff) > threshold) > 0:
        print "softmax check failed"
