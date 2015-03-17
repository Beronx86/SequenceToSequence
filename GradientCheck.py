__author__ = 'v-penlu'
import numpy as np
import EmoClassify_kmax as ECK
import EmoClassify as EC
import math
from numpy.linalg import norm
import SequenceToSequence as STS
import LanguageModel as LM
import os
real = np.float64

check_STS_2vocab = 0
check_STS_1vocab = 0
check_LM = 0
check_softmax = 0
check_Generage = 0
check_Beamsearch = 0
check_Cos = 0
check_Pool = 0
check_BLSTM_Cos = 0
check_KMax_pool = 0
check_KMax_BLSTM_Cos = 1
rng = np.random.RandomState()
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

if check_Cos:
    out_dir = "debug"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        files = os.listdir(out_dir)
        for f in files:
            os.remove(os.path.join(out_dir, f))
    vec_1 = rng.uniform(low=0, high=1, size=10)
    vec_2 = rng.uniform(low=0, high=1, size=10)
    _, Dl_1, Dl_2 = EC.Cos_feed_forward_backward(vec_1, vec_2, False)
    pertub = 1e-6
    thres = 1e-5
    vec_1_diff_m = np.zeros(vec_1.shape, dtype=real)
    vec_1_diff_a = np.zeros(vec_1.shape, dtype=real)
    for i in range(vec_1.shape[0]):
        vec_1[i] += pertub
        vec_1_diff_a[i], _, _ = EC.Cos_feed_forward_backward(vec_1, vec_2, False)
        vec_1[i] -= 2 * pertub
        vec_1_diff_m[i], _, _ = EC.Cos_feed_forward_backward(vec_1, vec_2, False)
        vec_1[i] += pertub
    numDl_1 = (vec_1_diff_a - vec_1_diff_m) / (2 * pertub)
    vec_2_diff_m = np.zeros(vec_2.shape, dtype=real)
    vec_2_diff_a = np.zeros(vec_2.shape, dtype=real)
    for i in range(vec_2.shape[0]):
        vec_2[i] += pertub
        vec_2_diff_a[i], _, _ = EC.Cos_feed_forward_backward(vec_1, vec_2, False)
        vec_2[i] -= 2 * pertub
        vec_2_diff_m[i], _, _ = EC.Cos_feed_forward_backward(vec_1, vec_2, False)
    numDl_2 = (vec_2_diff_a - vec_2_diff_m) / (2 * pertub)
    diff_1 = Dl_1 - numDl_1
    diff_2 = Dl_2 - numDl_2
    diff = [diff_1, diff_2]
    name = ["vec_1", "vec_2"]
    algDg = [Dl_1, Dl_2]
    numDg = [numDl_1, numDl_2]
    for i in range(2):
        if np.sum(np.abs(diff[i]) > thres) > 0:
            formatted_name = " " * (24 - len(name[i])) + name[i]
            print "%s\tgradient check failed" % formatted_name
            save_diff_name = "%s.failed.diff.txt" % name[i]
            np.savetxt(os.path.join("debug", save_diff_name), diff[i], "%+.6e")
            save_algDg_name = "%s.failed.algDg.txt" % name[i]
            np.savetxt(os.path.join("debug", save_algDg_name), algDg[i], "%+.6e")
            save_numDg_name = "%s.failed.numDg.txt" % name[i]
            np.savetxt(os.path.join("debug", save_numDg_name), numDg[i], "%+.6e")
        else:
            formatted_name = " " * (24 - len(name[i])) + name[i]
            print "%s\tgradient check succeeded" % formatted_name
            save_diff_name = "%s.succeeded.diff.txt" % name[i]
            np.savetxt(os.path.join("debug", save_diff_name), diff[i], "%+.6e")
            save_algDg_name = "%s.succeeded.algDg.txt" % name[i]
            np.savetxt(os.path.join("debug", save_algDg_name), algDg[i], "%+.6e")
            save_numDg_name = "%s.succeeded.numDg.txt" % name[i]
            np.savetxt(os.path.join("debug", save_numDg_name), numDg[i], "%+.6e")


def Pool_feed_forward_backward(in_seq_1, in_seq_2, weights_1, weights_2, is_pos,
                               pool_len=0, average=False):
    ts_1 = len(in_seq_1)
    ts_2 = len(in_seq_2)
    in_1 = []
    in_2 = []
    for i in range(ts_1):
        in_1.append(weights_1 * in_seq_1[i])
    for i in range(ts_2):
        in_2.append(weights_2 * in_seq_2[i])
    v_1, v_2, max_idx_1, max_idx_2 = EC.Pool_feed_forward(in_1, in_2, pool_len, average)
    loss, Dl_1, Dl_2 = EC.Cos_feed_forward_backward(v_1, v_2, is_pos)
    Dl_pool_1, Dl_pool_2 = EC.Pool_feed_backward(Dl_1, Dl_2, ts_1, ts_2, max_idx_1,
                                                 max_idx_2, pool_len, average)
    Dw_1 = np.zeros(weights_1.shape, dtype=real)
    Dw_2 = np.zeros(weights_2.shape, dtype=real)
    for i in range(ts_1):
        Dw_1 += Dl_pool_1[i] * in_seq_1[i]
    for i in range(ts_2):
        Dw_2 += Dl_pool_2[i] * in_seq_2[i]
    return loss, Dw_1, Dw_2

if check_Pool:
    out_dir = "debug"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        files = os.listdir(out_dir)
        for f in files:
            os.remove(os.path.join(out_dir, f))
    dim = 5
    len_1 = 17
    len_2 = 12
    seq_1 = []
    seq_2 = []
    is_pos = True
    pool_len = 3
    average = False
    for i in range(len_1):
        seq_1.append(rng.uniform(low=0, high=1, size=(dim, 1)))
    for i in range(len_2):
        seq_2.append(rng.uniform(low=0, high=1, size=(dim, 1)))
    weights_1 = rng.uniform(low=-0.1, high=0.1, size=(dim, 1))
    weights_2 = rng.uniform(low=-0.1, high=0.1, size=(dim, 1))
    loss, Dw_1, Dw_2 = Pool_feed_forward_backward(seq_1, seq_2, weights_1, weights_2,
                                                  is_pos, pool_len, average)
    pertub = 1e-6
    thres = 1e-5
    w_1_diff_m = np.zeros(weights_1.shape, dtype=real)
    w_1_diff_a = np.zeros(weights_1.shape, dtype=real)
    for i in range(weights_1.shape[0]):
        weights_1[i] += pertub
        w_1_diff_a[i], _, _ = Pool_feed_forward_backward(seq_1, seq_2, weights_1,
                                                         weights_2, is_pos, pool_len, average)
        weights_1[i] -= 2 * pertub
        w_1_diff_m[i], _, _ = Pool_feed_forward_backward(seq_1, seq_2, weights_1,
                                                         weights_2, is_pos, pool_len, average)
        weights_1[i] += pertub
    numDw_1 = (w_1_diff_a - w_1_diff_m) / (2 * pertub)
    w_2_diff_m = np.zeros(weights_2.shape, dtype=real)
    w_2_diff_a = np.zeros(weights_2.shape, dtype=real)
    for i in range(weights_2.shape[0]):
        weights_2[i] += pertub
        w_2_diff_a[i], _, _ = Pool_feed_forward_backward(seq_1, seq_2, weights_1,
                                                         weights_2, is_pos, pool_len, average)
        weights_2[i] -= 2 * pertub
        w_2_diff_m[i], _, _ = Pool_feed_forward_backward(seq_1, seq_2, weights_1,
                                                         weights_2, is_pos, pool_len, average)
        weights_2[i] += pertub
    numDw_2 = (w_2_diff_a - w_2_diff_m) / (2 * pertub)
    diff_1 = Dw_1 - numDw_1
    diff_2 = Dw_2 - numDw_2
    diff = [diff_1, diff_2]
    name = ["weights_1", "weights_2"]
    algDg = [Dw_1, Dw_2]
    numDg = [numDw_1, numDw_2]
    for i in range(2):
        if np.sum(np.abs(diff[i]) > thres) > 0:
            formatted_name = " " * (24 - len(name[i])) + name[i]
            print "%s\tgradient check failed" % formatted_name
            save_diff_name = "%s.failed.diff.txt" % name[i]
            np.savetxt(os.path.join("debug", save_diff_name), diff[i], "%+.6e")
            save_algDg_name = "%s.failed.algDg.txt" % name[i]
            np.savetxt(os.path.join("debug", save_algDg_name), algDg[i], "%+.6e")
            save_numDg_name = "%s.failed.numDg.txt" % name[i]
            np.savetxt(os.path.join("debug", save_numDg_name), numDg[i], "%+.6e")
        else:
            formatted_name = " " * (24 - len(name[i])) + name[i]
            print "%s\tgradient check succeeded" % formatted_name
            save_diff_name = "%s.succeeded.diff.txt" % name[i]
            np.savetxt(os.path.join("debug", save_diff_name), diff[i], "%+.6e")
            save_algDg_name = "%s.succeeded.algDg.txt" % name[i]
            np.savetxt(os.path.join("debug", save_algDg_name), algDg[i], "%+.6e")
            save_numDg_name = "%s.succeeded.numDg.txt" % name[i]
            np.savetxt(os.path.join("debug", save_numDg_name), numDg[i], "%+.6e")

if check_BLSTM_Cos:
    dim = 5
    len_1 = 17
    len_2 = 12
    seq_1 = []
    seq_2 = []
    is_pos = True
    pool_len = 3
    average = False
    for i in range(len_1):
        seq_1.append(rng.uniform(low=0, high=1, size=(dim, 1)))
    for i in range(len_2):
        seq_2.append(rng.uniform(low=0, high=1, size=(dim, 1)))
    hidden_size_list = [8, 9]
    mode = [True, 3, False]
    params = EC.Construct_net(hidden_size_list, dim)
    EC.Gradient_check(params, seq_1, seq_2, mode)


def KMax_pool_feed_forward_backward(in_seq_1, in_seq_2, k, weights_1, weights_2,
                                    is_pos):
    ts_1 = len(in_seq_1)
    ts_2 = len(in_seq_2)
    in_1 = []
    in_2 = []
    for i in range(ts_1):
        in_1.append(weights_1 * in_seq_1[i])
    for i in range(ts_2):
        in_2.append(weights_2 * in_seq_2[i])
    v_1, v_2, max_idx_1, max_idx_2 = EC.KMax_pool_feed_forward(in_1, in_2, k)
    loss, Dl_1, Dl_2 = EC.Cos_feed_forward_backward(v_1, v_2, is_pos)
    Dl_pool_1, Dl_pool_2 = EC.KMax_pool_feed_backward(Dl_1, Dl_2, ts_1, ts_2,
                                                      max_idx_1, max_idx_2, k)
    Dw_1 = np.zeros(weights_1.shape, dtype=real)
    Dw_2 = np.zeros(weights_2.shape, dtype=real)
    for i in range(ts_1):
        Dw_1 += Dl_pool_1[i] * in_seq_1[i]
    for i in range(ts_2):
        Dw_2 += Dl_pool_2[i] * in_seq_2[i]
    return loss, Dw_1, Dw_2

if check_KMax_pool:
    out_dir = "debug"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    else:
        files = os.listdir(out_dir)
        for f in files:
            os.remove(os.path.join(out_dir, f))
    dim = 5
    len_1 = 17
    len_2 = 12
    seq_1 = []
    seq_2 = []
    is_pos = True
    k = 3
    average = False
    for i in range(len_1):
        seq_1.append(rng.uniform(low=0, high=1, size=(dim, 1)))
    for i in range(len_2):
        seq_2.append(rng.uniform(low=0, high=1, size=(dim, 1)))
    weights_1 = rng.uniform(low=-0.1, high=0.1, size=(dim, 1))
    weights_2 = rng.uniform(low=-0.1, high=0.1, size=(dim, 1))
    loss, Dw_1, Dw_2 = KMax_pool_feed_forward_backward(seq_1, seq_2, k, weights_1,
                                                       weights_2, is_pos)
    pertub = 1e-6
    thres = 1e-5
    w_1_diff_m = np.zeros(weights_1.shape, dtype=real)
    w_1_diff_a = np.zeros(weights_1.shape, dtype=real)
    for i in range(weights_1.shape[0]):
        weights_1[i] += pertub
        w_1_diff_a[i], _, _ = KMax_pool_feed_forward_backward(seq_1, seq_2, k,
                                                              weights_1, weights_2, is_pos)
        weights_1[i] -= 2 * pertub
        w_1_diff_m[i], _, _ = KMax_pool_feed_forward_backward(seq_1, seq_2, k,
                                                              weights_1, weights_2, is_pos)
        weights_1[i] += pertub
    numDw_1 = (w_1_diff_a - w_1_diff_m) / (2 * pertub)
    w_2_diff_m = np.zeros(weights_2.shape, dtype=real)
    w_2_diff_a = np.zeros(weights_2.shape, dtype=real)
    for i in range(weights_2.shape[0]):
        weights_2[i] += pertub
        w_2_diff_a[i], _, _ = KMax_pool_feed_forward_backward(seq_1, seq_2, k,
                                                              weights_1, weights_2, is_pos)
        weights_2[i] -= 2 * pertub
        w_2_diff_m[i], _, _ = KMax_pool_feed_forward_backward(seq_1, seq_2, k,
                                                              weights_1, weights_2, is_pos)
        weights_2[i] += pertub
    numDw_2 = (w_2_diff_a - w_2_diff_m) / (2 * pertub)
    diff_1 = Dw_1 - numDw_1
    diff_2 = Dw_2 - numDw_2
    diff = [diff_1, diff_2]
    name = ["weights_1", "weights_2"]
    algDg = [Dw_1, Dw_2]
    numDg = [numDw_1, numDw_2]
    for i in range(2):
        if np.sum(np.abs(diff[i]) > thres) > 0:
            formatted_name = " " * (24 - len(name[i])) + name[i]
            print "%s\tgradient check failed" % formatted_name
            save_diff_name = "%s.failed.diff.txt" % name[i]
            np.savetxt(os.path.join("debug", save_diff_name), diff[i], "%+.6e")
            save_algDg_name = "%s.failed.algDg.txt" % name[i]
            np.savetxt(os.path.join("debug", save_algDg_name), algDg[i], "%+.6e")
            save_numDg_name = "%s.failed.numDg.txt" % name[i]
            np.savetxt(os.path.join("debug", save_numDg_name), numDg[i], "%+.6e")
        else:
            formatted_name = " " * (24 - len(name[i])) + name[i]
            print "%s\tgradient check succeeded" % formatted_name
            save_diff_name = "%s.succeeded.diff.txt" % name[i]
            np.savetxt(os.path.join("debug", save_diff_name), diff[i], "%+.6e")
            save_algDg_name = "%s.succeeded.algDg.txt" % name[i]
            np.savetxt(os.path.join("debug", save_algDg_name), algDg[i], "%+.6e")
            save_numDg_name = "%s.succeeded.numDg.txt" % name[i]
            np.savetxt(os.path.join("debug", save_numDg_name), numDg[i], "%+.6e")


if check_KMax_BLSTM_Cos:
    dim = 5
    len_1 = 17
    len_2 = 12
    seq_1 = []
    seq_2 = []
    is_pos = True
    k = 3
    average = False
    for i in range(len_1):
        seq_1.append(rng.uniform(low=0, high=1, size=(dim, 1)))
    for i in range(len_2):
        seq_2.append(rng.uniform(low=0, high=1, size=(dim, 1)))
    hidden_size_list = [8, 9]
    params, _ = ECK.Construct_net(hidden_size_list, dim, k)
    ECK.Gradient_check(params, seq_1, seq_2, is_pos)