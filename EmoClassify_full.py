__author__ = 'v-penlu'
import SequenceToSequence as STS
import EmoClassify as EC
import EmoClassify_kmax as ECK
import numpy as np
import os
import math
from numpy.linalg import norm
real = np.float64


def relu_prime(x):
    zero_idx = (x < 0)
    y = np.copy(x)
    y[zero_idx] = 0
    yp = np.ones(y.shape, dtype=real)
    yp[zero_idx] = 0
    return y, yp


def Full_feed_forward_single(weights, vec, act="logistic"):
    in_act = weights.dot(vec)
    if act == "relu":
        y, yp = relu_prime(in_act)
    elif act == "tanh":
        y, yp = STS.tanh_prime(in_act)
    elif act == "logistic":
        y, yp = STS.logistic_prime(in_act)
    else:
        print "activation func error"
        exit(1)
    return y, [yp, vec]


def Full_feed_backward_single(weights, in_errs, inter_vals):
    yp, vec = inter_vals
    Dl = in_errs * yp
    Dl = Dl.reshape(Dl.shape[0], 1)
    vec = vec.reshape(1, vec.shape[0])
    Dg = Dl.dot(vec)
    lower_input_err = weights.T.dot(Dl)
    return Dg, lower_input_err.reshape(-1)


def Full_feed_forward(weights, vec_1, vec_2, act="logistic"):
    y_1, inter_1 = Full_feed_forward_single(weights, vec_1, act)
    y_2, inter_2 = Full_feed_forward_single(weights, vec_2, act)
    return y_1, y_2, inter_1, inter_2


def Full_feed_backward(weights, in_err_1, in_err_2, inter_1, inter_2):
    Dg_1, lower_input_errs_1 = Full_feed_backward_single(weights, in_err_1,
                                                         inter_1)
    Dg_2, lower_input_errs_2 = Full_feed_backward_single(weights, in_err_2,
                                                         inter_2)
    Dg_1 += Dg_2
    return Dg_1, lower_input_errs_1, lower_input_errs_2


def Out_feed_forward_backward_kmax(weights, in_seq_1, in_seq_2, is_pos, k,
                                   act='logistic'):
    ts_1 = len(in_seq_1)
    ts_2 = len(in_seq_2)
    v_1, v_2, max_idx_1, max_idx_2 = ECK.KMax_pool_feed_forward(in_seq_1,
                                                                in_seq_2, k)
    fv_1, fv_2, finter_1, finter_2 = Full_feed_forward(weights, v_1, v_2, act)
    loss, Dl_1, Dl_2 = EC.Cos_feed_forward_backward(fv_1, fv_2, is_pos)
    Dg, Dl_full_1, Dl_full_2 = Full_feed_backward(weights, Dl_1, Dl_2, finter_1,
                                                  finter_2)
    Dl_pool_1, Dl_pool_2 = ECK.KMax_pool_feed_backward(Dl_full_1, Dl_full_2, ts_1, ts_2,
                                                       max_idx_1, max_idx_2, k)
    return loss, Dg, Dl_pool_1, Dl_pool_2


def Feed_forward_backward_kmax(params, in_seq_1, in_seq_2, is_pos, mode=0):
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
    loss, Dg, Dl_out_s1, Dl_out_s2 = Out_feed_forward_backward_kmax(
        params["out"], lower_acts_1, lower_acts_2, is_pos, params["k"],
        params["act"])
    if mode == 1:
        return loss
    grads["out"] = Dg
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


def Out_feed_forward_backward_lmax(weights, in_seq_1, in_seq_2, is_pos, pool_len,
                                   average=False, act='logistic'):
    ts_1 = len(in_seq_1)
    ts_2 = len(in_seq_2)
    v_1, v_2, max_idx_1, max_idx_2 = EC.Pool_feed_forward(in_seq_1, in_seq_2,
                                                          pool_len, average)
    fv_1, fv_2, finter_1, finter_2 = Full_feed_forward(weights, v_1, v_2, act)
    loss, Dl_1, Dl_2 = EC.Cos_feed_forward_backward(fv_1, fv_2, is_pos)
    Dg, Dl_full_1, Dl_full_2 = Full_feed_backward(weights, Dl_1, Dl_2, finter_1,
                                                  finter_2)
    Dl_pool_1, Dl_pool_2 = EC.Pool_feed_backward(Dl_full_1, Dl_full_2, ts_1,
                                                 ts_2, max_idx_1, max_idx_2,
                                                 pool_len, average)
    return loss, Dg, Dl_pool_1, Dl_pool_2


def Feed_forward_backward_lmax(params, in_seq_1, in_seq_2, is_pos, mode=0):
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
    loss, Dg, Dl_out_s1, Dl_out_s2 = Out_feed_forward_backward_lmax(
        params["out"], lower_acts_1, lower_acts_2, is_pos, params["pool_len"],
        params["average"], params["act"])
    if mode == 1:
        return loss
    grads["out"] = Dg
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


def Construct_net(hidden_size_list, in_size, out_dim, pool_mode, act,
                  init_range=0.1, sgd_mode="ada"):
    rng = np.random.RandomState()
    params = dict()
    grad_acc = dict()
    params["num_layers"] = len(hidden_size_list)
    params["act"] = act
    if len(pool_mode) == 2:
        params["pool_mode"] = "lmax"
        params["pool_len"] = pool_mode[0]
        params["average"] = pool_mode[1]
    else:
        params["pool_mode"] = "kmax"
        params["k"] = pool_mode[0]
    input_size = in_size
    params["out"] = np.asarray(rng.uniform(low=-init_range, high=init_range,
                                           size=(out_dim, 2 * pool_mode[0] * hidden_size_list[-1])), dtype=real)
    if sgd_mode.startswith("ada"):
        grad_acc["out"] = np.ones(params["out"].shape, dtype=real)
    else:
        grad_acc["out"] = np.zeros(params["out"].shape, dtype=real)
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
        if sgd_mode.startswith("ada"):
            for W in params[f_layer_name]:
                grad_acc[f_layer_name].append(np.ones(W.shape, dtype=real))
            for W in params[b_layer_name]:
                grad_acc[b_layer_name].append(np.ones(W.shape, dtype=real))
        else:
            for W in params[f_layer_name]:
                grad_acc[f_layer_name].append(np.zeros(W.shape, dtype=real))
            for W in params[b_layer_name]:
                grad_acc[b_layer_name].append(np.zeros(W.shape, dtype=real))
    return params, grad_acc


def Feed_forward_backward(params, seq_1, seq_2, is_pos, mode=0):
    if params["pool_mode"] == "kmax":
        return Feed_forward_backward_kmax(params, seq_1, seq_2, is_pos, mode)
    else:
        return Feed_forward_backward_lmax(params, seq_1, seq_2, is_pos, mode)


def Pool_feed_forward_single(params, in_seq):
    if params["pool_mode"] == "lmax":
        vec = EC.Pool_feed_forward_single(in_seq, params["pool_len"],
                                          params["average"])
    else:
        vec, _ = ECK.KMax_pool_feed_forward_single(in_seq, params["k"])
    return vec


def Extract_feature(params, in_seq):
    lower_acts = in_seq
    for i in range(params["num_layers"]):
        f_layer_name = "LSTM_layer_f" + str(i)
        b_layer_name = "LSTM_layer_b" + str(i)
        ret = EC.Bi_LSTM_feed_forward(params[f_layer_name], params[b_layer_name],
                                      lower_acts)
        lower_acts = ret[0]
    pooled_vec = Pool_feed_forward_single(params, lower_acts)
    ret_vec, _ = Full_feed_forward_single(params["out"], pooled_vec, params["act"])
    return ret_vec


def Gradient_check(params, seq_1, seq_2, is_pos):
    if not os.path.exists("debug"):
        os.mkdir("debug")
    else:
        files = os.listdir("debug")
        for f in files:
            os.remove(os.path.join("debug", f))
    print "start gradient check"
    grads, _ = Feed_forward_backward(params, seq_1, seq_2, is_pos)
    Check_diff(params, grads, "out", seq_1, seq_2, is_pos)
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


def All_params_SGD(params, grads, learn_rate=0.1, clip_norm=5, mode="ada",
                   grad_acc=0, momentum=0.95):
    if mode not in ["sgd", "momentum", "ada", "adaautocoor"]:
        mode = "ada"
    EC.SGD(params["out"], grads["out"], mode, learn_rate, momentum,
           grad_acc["out"], clip_norm)
    for i in range(params["num_layers"]):
        f_layer_name = "LSTM_layer_f" + str(i)
        b_layer_name = "LSTM_layer_b" + str(i)
        for W, g, g_a in zip(params[f_layer_name], grads[f_layer_name],
                             grad_acc[f_layer_name]):
            EC.SGD(W, g, mode, learn_rate, momentum, g_a, clip_norm)
        for W, g, g_a in zip(params[b_layer_name], grads[b_layer_name],
                             grad_acc[f_layer_name]):
            EC.SGD(W, g, mode, learn_rate, momentum, g_a, clip_norm)
