__author__ = 'v-penlu'
import numpy as np
import EmoClassify as EC
import os
import math
from numpy.linalg import norm
real = np.float64


def KMax_pool_feed_forward_single(input_seq, k):
    feature_dim = input_seq[0].shape[0]
    feature_mtx = np.zeros((len(input_seq), feature_dim), dtype=real)
    feature_vec = np.zeros((feature_dim * k), dtype=real)
    max_idx = []
    for i, v in enumerate(input_seq):
        feature_mtx[i] = input_seq[i][:, 0]
    order_mtx = np.argsort(feature_mtx, axis=0)
    for i in range(feature_dim):
        s = i * k
        e = s + k
        order = order_mtx[:, i]
        feat = feature_mtx[:, i]
        max_k_idx = order[-k:]
        dec_max_k_idx = np.sort(max_k_idx)
        feature_vec[s:e] = feat[dec_max_k_idx]
        max_idx.append(dec_max_k_idx)
    return feature_vec, max_idx


def KMax_pool_feed_backward_single(input_errs, time_steps, max_idx, k):
    feature_dim = input_errs.shape[0] / k
    Dl_mtx = np.zeros((time_steps, feature_dim), dtype=real)
    for i in range(feature_dim):
        s = k * i
        e = s + k
        r_idx = max_idx[i]
        Dl_mtx[r_idx, i] = input_errs[s:e]
    Dl_pool = [Dl_mtx[i].reshape(feature_dim, 1) for i in range(time_steps)]
    return Dl_pool


def KMax_pool_feed_forward(in_seq_1, in_seq_2, k):
    vec_1, max_idx_1 = KMax_pool_feed_forward_single(in_seq_1, k)
    vec_2, max_idx_2 = KMax_pool_feed_forward_single(in_seq_2, k)
    return vec_1, vec_2, max_idx_1, max_idx_2


def KMax_pool_feed_backward(in_err_1, in_err_2, time_steps_1, time_steps_2,
                            max_idx_1, max_idx_2, k):
    Dl_pool_1 = KMax_pool_feed_backward_single(in_err_1, time_steps_1,
                                               max_idx_1, k)
    Dl_pool_2 = KMax_pool_feed_backward_single(in_err_2, time_steps_2,
                                               max_idx_2, k)
    return Dl_pool_1, Dl_pool_2


def Out_feed_forward_backward(in_seq_1, in_seq_2, is_pos, k):
    ts_1 = len(in_seq_1)
    ts_2 = len(in_seq_2)
    v_1, v_2, max_idx_1, max_idx_2 = KMax_pool_feed_forward(in_seq_1, in_seq_2,
                                                            k)
    loss, Dl_1, Dl_2 = EC.Cos_feed_forward_backward(v_1, v_2, is_pos)
    Dl_pool_1, Dl_pool_2 = KMax_pool_feed_backward(Dl_1, Dl_2, ts_1, ts_2,
                                                   max_idx_1, max_idx_2, k)
    return loss, Dl_pool_1, Dl_pool_2


def Construct_net(hidden_size_list, in_size, k, init_range=0.1,
                       mode="ada"):
    rng = np.random.RandomState()
    params = dict()
    grad_acc = dict()
    params["num_layers"] = len(hidden_size_list)
    params["k"] = k
    input_size = in_size
    for i, layer_size in enumerate(hidden_size_list):
        f_layer_name = "LSTM_layer_f" + str(i)
        b_layer_name = "LSTM_layer_b" + str(i)
        joint_size = layer_size + input_size + 1
        # init forward weights
        W_iota_y = np.asarray(rng.uniform(low=-init_range, high=init_range,
                                          size=(layer_size, joint_size)), dtype=real)
        W_iota_s = np.asarray(rng.uniform(low=-init_range, high=init_range,
                                          size=(layer_size, 1)), dtype=real)
        W_phi_y = np.asarray(rng.uniform(low=-init_range, high=init_range,
                                         size=(layer_size, joint_size)), dtype=real)
        W_phi_s = np.asarray(rng.uniform(low=-init_range, high=init_range,
                                         size=(layer_size, 1)), dtype=real)
        W = np.asarray(rng.uniform(low=-init_range, high=init_range,
                                   size=(layer_size, joint_size)), dtype=real)
        W_eta_y = np.asarray(rng.uniform(low=-init_range, high=init_range,
                                         size=(layer_size, joint_size)), dtype=real)
        W_eta_s = np.asarray(rng.uniform(low=-init_range, high=init_range,
                                         size=(layer_size, 1)), dtype=real)
        params[f_layer_name] = [W_iota_y, W_iota_s, W_phi_y, W_phi_s, W, W_eta_y, W_eta_s]
        # init backward weights
        W_iota_y = np.asarray(rng.uniform(low=-init_range, high=init_range,
                                          size=(layer_size, joint_size)), dtype=real)
        W_iota_s = np.asarray(rng.uniform(low=-init_range, high=init_range,
                                          size=(layer_size, 1)), dtype=real)
        W_phi_y = np.asarray(rng.uniform(low=-init_range, high=init_range,
                                         size=(layer_size, joint_size)), dtype=real)
        W_phi_s = np.asarray(rng.uniform(low=-init_range, high=init_range,
                                         size=(layer_size, 1)), dtype=real)
        W = np.asarray(rng.uniform(low=-init_range, high=init_range,
                                   size=(layer_size, joint_size)), dtype=real)
        W_eta_y = np.asarray(rng.uniform(low=-init_range, high=init_range,
                                         size=(layer_size, joint_size)), dtype=real)
        W_eta_s = np.asarray(rng.uniform(low=-init_range, high=init_range,
                                         size=(layer_size, 1)), dtype=real)
        params[b_layer_name] = [W_iota_y, W_iota_s, W_phi_y, W_phi_s, W, W_eta_y, W_eta_s]
        input_size = layer_size * 2
        grad_acc[f_layer_name] = []
        grad_acc[b_layer_name] = []
        if mode.startswith("ada"):
            for W in params[f_layer_name]:
                grad_acc[f_layer_name].append(np.ones(W.shape, dtype=real))
            for W in params[b_layer_name]:
                grad_acc[b_layer_name].append(np.ones(W.shape, dtype=real))
        elif mode == "momentum":
            for W in params[f_layer_name]:
                grad_acc[f_layer_name].append(np.zeros(W.shape, dtype=real))
            for W in params[b_layer_name]:
                grad_acc[b_layer_name].append(np.zeros(W.shape, dtype=real))
    if mode == "sgd":
        grad_acc = 0
    return params, grad_acc


def Feed_forward_backward(params, in_seq_1, in_seq_2, is_pos, mode=0):
    """
    :type params: dict
    :type in_seq_1: list
    :type in_seq_2: list
    :type is_pos: bool
    :type pool_len: int
    :type avg: bool
    :param mode:  mode=0 feed forward and backward
                  mode=1 feed forward for valid and test samples
    :return:
    """
    inter_vals = dict()
    grads = dict()
    # feed forward seq 1 & 2
    lower_acts_1 = in_seq_1
    lower_acts_2 = in_seq_2
    for i in range(params["num_layers"]):
        f_layer_name = "LSTM_layer_f" + str(i)
        b_layer_name = "LSTM_layer_b" + str(i)
        ret_1 = EC.Bi_LSTM_feed_forward(params[f_layer_name], params[b_layer_name],
                                        lower_acts_1)
        ret_2 = EC.Bi_LSTM_feed_forward(params[f_layer_name], params[b_layer_name],
                                        lower_acts_2)
        inter_vals[f_layer_name + "s1"] = ret_1[1]
        inter_vals[b_layer_name + "s1"] = ret_1[2]
        inter_vals[f_layer_name + "s2"] = ret_2[1]
        inter_vals[b_layer_name + "s2"] = ret_2[2]
        lower_acts_1 = ret_1[0]
        lower_acts_2 = ret_2[0]
    loss, Dl_out_s1, Dl_out_s2 = Out_feed_forward_backward(lower_acts_1, lower_acts_2,
                                                                is_pos, params["k"])
    if mode == 1:
        return loss
    in_err_1 = Dl_out_s1
    in_err_2 = Dl_out_s2
    for i in reversed(range(params["num_layers"])):
        f_layer_name = "LSTM_layer_f" + str(i)
        b_layer_name = "LSTM_layer_b" + str(i)
        f_iv_s1 = inter_vals[f_layer_name + "s1"]
        b_iv_s1 = inter_vals[b_layer_name + "s1"]
        f_iv_s2 = inter_vals[f_layer_name + "s2"]
        b_iv_s2 = inter_vals[b_layer_name + "s2"]
        ret_1 = EC.Bi_LSTM_feed_backward(params[f_layer_name], params[b_layer_name],
                                         f_iv_s1, b_iv_s1, in_err_1)
        ret_2 = EC.Bi_LSTM_feed_backward(params[f_layer_name], params[b_layer_name],
                                         f_iv_s2, b_iv_s2, in_err_2)
        grads[f_layer_name] = [x + y for (x, y) in zip(ret_1[0], ret_2[0])]
        grads[b_layer_name] = [x + y for (x, y) in zip(ret_1[1], ret_2[1])]
        in_err_1 = ret_1[2]
        in_err_2 = ret_2[2]
    return grads, loss


def Extract_feature(params, in_seq):
    lower_acts = in_seq
    for i in range(params["num_layers"]):
        f_layer_name = "LSTM_layer_f" + str(i)
        b_layer_name = "LSTM_layer_b" + str(i)
        ret = EC.Bi_LSTM_feed_forward(params[f_layer_name], params[b_layer_name],
                                      lower_acts)
        lower_acts = ret[0]
    feature_vec = KMax_pool_feed_forward_single(lower_acts, params["k"])
    return feature_vec



def Gradient_check(params, seq_1, seq_2, is_pos):
    if not os.path.exists("debug"):
        os.mkdir("debug")
    else:
        files = os.listdir("debug")
        for f in files:
            os.remove(os.path.join("debug", f))
    print "start gradient check"
    grads, _ = Feed_forward_backward(params, seq_1, seq_2, is_pos)
    for i in range(params["num_layers"]):
        f_layer_name = "LSTM_layer_f" + str(i)
        b_layer_name = "LSTM_layer_b" + str(i)
        Check_diff(params, grads, f_layer_name, seq_1, seq_2, is_pos)
        Check_diff(params, grads, b_layer_name, seq_1, seq_2, is_pos)


def Auto_grad(params, fluct_weights, seq_1, seq_2, is_pos):
    """This function refers to minFunc autoGrad function
    :param params:
    :param fluct_weights: A weight matrix in params. It's ptr, change the
           element here, corresponding elements in params wil also be changed.
    :param sample:
    :return numerical_grad
    """
    W = fluct_weights
    perturbation = 2 * math.sqrt(1e-12) * (1 + norm(W, 2))
    diff1 = np.zeros(W.shape, dtype=real)
    diff2 = np.zeros(W.shape, dtype=real)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] += perturbation
            diff1[i, j] = Feed_forward_backward(params, seq_1, seq_2, is_pos,
                                                mode=1)
            W[i, j] -= 2 * perturbation
            diff2[i, j] = Feed_forward_backward(params, seq_1, seq_2, is_pos,
                                                mode=1)
            # Restore the weight value at (i,j)
            W[i, j] += perturbation
    numerical_grad = (diff1 - diff2) / (2 * perturbation)
    return numerical_grad


def Check_diff(params, grads, name, seq_1, seq_2, is_pos):
    """
    :param params:
    :param grads:
    :param name:
    :param sample:
    :return:
    """
    weights_name = ["W_iota_y", "W_iota_s", "W_phi_y", "W_phi_s", "W",
                    "W_eta_y", "W_eta_s"]
    sigFigs = 6
    if isinstance(params[name], list):
        for i, weights in enumerate(params[name]):
            numDg = Auto_grad(params, weights, seq_1, seq_2, is_pos)
            algDg = grads[name][i]
            diff = algDg - numDg
            minDg = np.minimum(np.abs(algDg), np.abs(numDg))
            minDg[minDg == 0] = 1
            threshold = np.power(10.0, np.maximum(0.0, np.ceil(np.log10(minDg)))-int(sigFigs))
            if np.sum(np.abs(diff) > threshold) > 0:
                temp = name + "_" + weights_name[i]
                formatted_name = " " * (24 - len(temp)) + temp
                print "%s\tgradient check failed" % formatted_name
                save_name = "%s_%s.filed.diff.txt" % (name, weights_name[i])
                np.savetxt(os.path.join("debug", save_name), diff, "%+.6e")
                save_algDg_name = "%s_%s.failed.algDg.txt" % (name, weights_name[i])
                np.savetxt(os.path.join("debug", save_algDg_name), algDg, "%+.6e")
                save_numDg_name = "%s_%s.failed.numDg.txt" % (name, weights_name[i])
                np.savetxt(os.path.join("debug", save_numDg_name), numDg, "%+.6e")
            else:
                temp = name + "_" + weights_name[i]
                formatted_name = " " * (24 - len(temp)) + temp
                print "%s\tgradient check succeeded" % formatted_name
                save_name = "%s_%s.succeeded.diff.txt" % (name, weights_name[i])
                np.savetxt(os.path.join("debug", save_name), diff, "%+.6e")
                save_algDg_name = "%s_%s.succeeded.algDg.txt" % (name, weights_name[i])
                np.savetxt(os.path.join("debug", save_algDg_name), algDg, "%+.6e")
                save_numDg_name = "%s_%s.succeeded.numDg.txt" % (name, weights_name[i])
                np.savetxt(os.path.join("debug", save_numDg_name), numDg, "%+.6e")
    else:
        numDg = Auto_grad(params, params[name], seq_1, seq_2, is_pos)
        algDg = grads[name]
        diff = algDg - numDg
        minDg = np.minimum(np.abs(algDg), np.abs(numDg))
        minDg[minDg == 0] = 1
        threshold = np.power(10.0, np.maximum(0.0, np.ceil(np.log10(minDg)))-int(sigFigs))
        if np.sum(np.abs(diff) > threshold) > 0:
            formatted_name = " " * (24 - len(name)) + name
            print "%s\tgradient check failed" % formatted_name
            save_diff_name = "%s.failed.diff.txt" % name
            np.savetxt(os.path.join("debug", save_diff_name), diff, "%+.6e")
            save_algDg_name = "%s.failed.algDg.txt" % name
            np.savetxt(os.path.join("debug", save_algDg_name), algDg, "%+.6e")
            save_numDg_name = "%s.failed.numDg.txt" % name
            np.savetxt(os.path.join("debug", save_numDg_name), numDg, "%+.6e")
        else:
            formatted_name = " " * (24 - len(name)) + name
            print "%s\tgradient check succeeded" % formatted_name
            save_diff_name = "%s.succeeded.diff.txt" % name
            np.savetxt(os.path.join("debug", save_diff_name), diff, "%+.6e")
            save_algDg_name = "%s.succeeded.algDg.txt" % name
            np.savetxt(os.path.join("debug", save_algDg_name), algDg, "%+.6e")
            save_numDg_name = "%s.succeeded.numDg.txt" % name
            np.savetxt(os.path.join("debug", save_numDg_name), numDg, "%+.6e")

