__author__ = 'Liu'
import numpy as np
import SequenceToSequence as STS
from scipy.spatial.distance import cosine
import math
import operator
import os
from numpy.linalg import norm
real = np.float64


def accumulate(num_list):
    acc_list = []
    for i in range(len(num_list)):
        acc_list.append(sum(num_list[:i + 1]))
    return acc_list


def Calculate_frame_idx(time_steps, pool_len):
    num_frames = [time_steps / pool_len] * pool_len
    num_frames[-1] += time_steps % pool_len
    frame_idx = accumulate(num_frames)
    frame_idx.insert(0, 0)
    return frame_idx


def Pool_feed_forward(in_seq_1, in_seq_2, pool_len=0, average=True):
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
    max_idx_1 = []
    max_idx_2 = []
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
        feature_vec_1 = feature_vec_1.reshape(-1)
        feature_vec_2 = feature_vec_2.reshape(-1)
    else:
        feature_dim = in_seq_1[0].shape[0]
        frame_idx_1 = Calculate_frame_idx(len(in_seq_1), pool_len)
        frame_idx_2 = Calculate_frame_idx(len(in_seq_2), pool_len)
        feature_mtx_1 = np.zeros((len(in_seq_1), feature_dim), dtype=real)
        feature_mtx_2 = np.zeros((len(in_seq_2), feature_dim), dtype=real)
        for i, v in enumerate(in_seq_1):
            feature_mtx_1[i] = in_seq_1[i][:, 0]
        for i, v in enumerate(in_seq_2):
            feature_mtx_2[i] = in_seq_2[i][:, 0]
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
                max_idx_1.append(np.argmax(feature_mtx_1[frame_idx_1[i]: frame_idx_1[i + 1]],
                                           axis=0) + frame_idx_1[i])
                max_idx_2.append(np.argmax(feature_mtx_2[frame_idx_2[i]: frame_idx_2[i + 1]],
                                           axis=0) + frame_idx_2[i])
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
                    Dl_pool_1[j] = (input_errs_1[s: e].reshape(feature_dim, 1) /
                                    (frame_idx_1[i + 1] - frame_idx_1[i]))
                for j in range(frame_idx_2[i], frame_idx_2[i + 1]):
                    Dl_pool_2[j] = (input_errs_2[s: e].reshape(feature_dim, 1) /
                                    (frame_idx_2[i + 1] - frame_idx_2[i]))
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


def Cos_feed_forward_backward(vec_1, vec_2, is_posative):
    vec_1_pow = vec_1.dot(vec_1)
    vec_2_pow = vec_2.dot(vec_2)
    vec_1_norm = math.sqrt(vec_1_pow)
    vec_2_norm = math.sqrt(vec_2_pow)
    vec_1_2_p = vec_1.dot(vec_2)
    cos = vec_1_2_p / (vec_1_norm * vec_2_norm)
    # cos = 1 - cosine(vec_1, vec_2)
    DCos_1 = vec_2 / (vec_1_norm * vec_2_norm) - vec_1 * cos / vec_1_pow
    DCos_2 = vec_1 / (vec_1_norm * vec_2_norm) - vec_2 * cos / vec_2_pow
    if is_posative:
        loss = (1 - cos) / 2
        Dl_1 = -0.5 * DCos_1
        Dl_2 = -0.5 * DCos_2
    else:
        loss = cos * cos
        Dl_1 = 2 * cos * DCos_1
        Dl_2 = 2 * cos * DCos_2
    return loss, Dl_1, Dl_2


def Out_feed_forward_backward(in_seq_1, in_seq_2, is_pos, pool_len=0, average=False):
    ts_1 = len(in_seq_1)
    ts_2 = len(in_seq_2)
    v_1, v_2, max_idx_1, max_idx_2 = Pool_feed_forward(in_seq_1, in_seq_2,
                                                       pool_len, average)
    loss, Dl_1, Dl_2 = Cos_feed_forward_backward(v_1, v_2, is_pos)
    Dl_pool_1, Dl_pool_2 = Pool_feed_backward(Dl_1, Dl_2, ts_1, ts_2, max_idx_1,
                                              max_idx_2, pool_len, average)
    return loss, Dl_pool_1, Dl_pool_2


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
    sum_errs = []
    for f_errs, b_errs in zip(f_lower_input_errs, reversed(b_lower_input_errs)):
        sum_errs.append(f_errs + b_errs)
    return sum_errs


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


def Construct_net(hidden_size_list, in_size, pool_len, avg, init_range=0.1,
                  mode="ada"):
    rng = np.random.RandomState()
    params = dict()
    grad_acc = dict()
    params["num_layers"] = len(hidden_size_list)
    params["pool_len"] = pool_len
    params["average"] = avg
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
        else:
            for W in params[f_layer_name]:
                grad_acc[f_layer_name].append(np.zeros(W.shape, dtype=real))
            for W in params[b_layer_name]:
                grad_acc[b_layer_name].append(np.zeros(W.shape, dtype=real))
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
        ret_1 = Bi_LSTM_feed_forward(params[f_layer_name], params[b_layer_name],
                                     lower_acts_1)
        ret_2 = Bi_LSTM_feed_forward(params[f_layer_name], params[b_layer_name],
                                     lower_acts_2)
        inter_vals[f_layer_name + "s1"] = ret_1[1]
        inter_vals[b_layer_name + "s1"] = ret_1[2]
        inter_vals[f_layer_name + "s2"] = ret_2[1]
        inter_vals[b_layer_name + "s2"] = ret_2[2]
        lower_acts_1 = ret_1[0]
        lower_acts_2 = ret_2[0]
    loss, Dl_out_s1, Dl_out_s2 = Out_feed_forward_backward(lower_acts_1, lower_acts_2,
                                                           is_pos, params["pool_len"],
                                                           params["average"])
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
        ret_1 = Bi_LSTM_feed_backward(params[f_layer_name], params[b_layer_name],
                                      f_iv_s1, b_iv_s1, in_err_1)
        ret_2 = Bi_LSTM_feed_backward(params[f_layer_name], params[b_layer_name],
                                      f_iv_s2, b_iv_s2, in_err_2)
        grads[f_layer_name] = [x + y for (x, y) in zip(ret_1[0], ret_2[0])]
        grads[b_layer_name] = [x + y for (x, y) in zip(ret_1[1], ret_2[1])]
        in_err_1 = ret_1[2]
        in_err_2 = ret_2[2]
    return grads, loss


def All_params_SGD(params, grads, learn_rate=0.1, clip_norm=5, mode="ada",
                   grad_acc=0, momentum=0.95):
    if mode not in ["sgd", "momentum", "ada", "adaautocoor"]:
        mode = "ada"
    for i in range(params["num_layers"]):
        f_layer_name = "LSTM_layer_f" + str(i)
        b_layer_name = "LSTM_layer_b" + str(i)
        for W, g, g_a in zip(params[f_layer_name], grads[f_layer_name],
                             grad_acc[f_layer_name]):
            SGD(W, g, mode, learn_rate, momentum, g_a, clip_norm)
        for W, g, g_a in zip(params[b_layer_name], grads[b_layer_name],
                             grad_acc[f_layer_name]):
            SGD(W, g, mode, learn_rate, momentum, g_a, clip_norm)


def SGD(weight, gradient, mode, learn_rate=0.1, momentum=0.95, grad_acc=0,
        clip_norm=5):
    if mode == "sgd":
        STS.Weight_SGD(weight, gradient, learn_rate, clip_norm)
    elif mode == "momentum":
        Weight_SGD_mnt(weight, gradient, grad_acc, learn_rate, momentum,
                       clip_norm)
    elif mode == "ada":
        Weight_SGD_ada(weight, gradient, grad_acc, learn_rate, clip_norm)
    else:
        Weight_SGD_adaautocoor(weight, gradient, grad_acc, learn_rate, momentum,
                               clip_norm)



def Weight_SGD_mnt(weight, gradient, gradmnt, learn_rate=0.1, momentum=0.95,
                   clip_norm=5):
    if clip_norm > 0:
        grads_norm = gradient * gradient
        clip_idx = grads_norm > clip_norm * clip_norm
        gradient[clip_idx] = clip_norm * gradient[clip_idx] / grads_norm[clip_idx]
    if np.sum(gradmnt == 0) == gradient.size:
        gradmnt = gradient
    else:
        gradmnt = momentum * gradmnt + (1 - momentum) * gradient
    weight -= learn_rate * gradmnt


def Weight_SGD_ada(weight, gradient, gradsq, learn_rate=0.1, clip_norm=5):
    fudge_factor = 1e-6     # for numerical stability
    if clip_norm > 0:
        grads_norm = gradient * gradient
        clip_idx = grads_norm > clip_norm * clip_norm
        gradient[clip_idx] = clip_norm * gradient[clip_idx] / grads_norm[clip_idx]
    gradsq += gradient * gradient
    gradient /= (fudge_factor + np.sqrt(gradsq))
    weight -= learn_rate * gradient


def Weight_SGD_adaautocoor(weight, gradient, gradsq, learn_rate=0.1,
                           autocorr=0.95, clip_norm=5):
    fudge_factor = 1e-6     # for numerical stability
    if clip_norm > 0:
        grads_norm = gradient * gradient
        clip_idx = grads_norm > clip_norm * clip_norm
        gradient[clip_idx] = clip_norm * gradient[clip_idx] / grads_norm[clip_idx]
    gradsq += autocorr * gradient + (1 - autocorr) * gradient * gradient
    gradient /= (fudge_factor + np.sqrt(gradsq))
    weight -= learn_rate * gradient


def Gradient_check(params, seq_1, seq_2, mode):
    if not os.path.exists("debug"):
        os.mkdir("debug")
    else:
        files = os.listdir("debug")
        for f in files:
            os.remove(os.path.join("debug", f))
    print "start gradient check"
    grads, _ = Feed_forward_backward(params, seq_1, seq_2, mode[0], mode[1], mode[2])
    for i in range(params["num_layers"]):
        f_layer_name = "LSTM_layer_f" + str(i)
        b_layer_name = "LSTM_layer_b" + str(i)
        Check_diff(params, grads, f_layer_name, seq_1, seq_2, mode)
        Check_diff(params, grads, b_layer_name, seq_1, seq_2, mode)


def Auto_grad(params, fluct_weights, seq_1, seq_2, mode):
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
            _, diff1[i, j] = Feed_forward_backward(params, seq_1, seq_2, mode[0],
                                                   mode[1], mode[2])
            W[i, j] -= 2 * perturbation
            _, diff2[i, j] = Feed_forward_backward(params, seq_1, seq_2, mode[0],
                                                   mode[1], mode[2])
            # Restore the weight value at (i,j)
            W[i, j] += perturbation
    numerical_grad = (diff1 - diff2) / (2 * perturbation)
    return numerical_grad


def Check_diff(params, grads, name, seq_1, seq_2, mode):
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
            numDg = Auto_grad(params, weights, seq_1, seq_2, mode)
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
        numDg = Auto_grad(params, params[name], seq_1, seq_2, mode)
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


def Pool_feed_forward_single(in_seq_1, pool_len, average):
    max_idx_1 = []
    if pool_len == 0:
        feature_vec_1 = 0
        for v in in_seq_1:
            feature_vec_1 += v
        if average:
            feature_vec_1 /= len(in_seq_1)
        feature_vec_1 = feature_vec_1.reshape(-1)
    else:
        feature_dim = in_seq_1[0].shape[0]
        frame_idx_1 = Calculate_frame_idx(len(in_seq_1), pool_len)
        feature_mtx_1 = np.zeros((len(in_seq_1), feature_dim), dtype=real)
        for i, v in enumerate(in_seq_1):
            feature_mtx_1[i] = in_seq_1[i][:, 0]
        feature_vec_1 = np.zeros((feature_dim * pool_len), dtype=real)
        if average:
            for i in range(pool_len):
                s = feature_dim * i
                e = s + feature_dim
                feature_vec_1[s: e] = np.average(feature_mtx_1[frame_idx_1[i]: frame_idx_1[i + 1]], axis=0)
        else:
            for i in range(pool_len):
                s = feature_dim * i
                e = s + feature_dim
                feature_vec_1[s: e] = np.max(feature_mtx_1[frame_idx_1[i]: frame_idx_1[i + 1]], axis=0)
                # In max pooling back propagation, only errors only propagate to
                # max units.
                max_idx_1.append(np.argmax(feature_mtx_1[frame_idx_1[i]: frame_idx_1[i + 1]],
                                           axis=0) + frame_idx_1[i])
    return feature_vec_1, max_idx_1


def Extract_feature(params, in_seq, pool_len, avg=False):
    lower_acts = in_seq
    for i in range(params["num_layers"]):
        f_layer_name = "LSTM_layer_f" + str(i)
        b_layer_name = "LSTM_layer_b" + str(i)
        ret = Bi_LSTM_feed_forward(params[f_layer_name], params[b_layer_name],
                                   lower_acts)
        lower_acts = ret[0]
    feature_vec = Pool_feed_forward_single(lower_acts, pool_len, avg)
    return feature_vec
