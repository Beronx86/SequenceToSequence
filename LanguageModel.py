__author__ = 'v-penlu'
import SequenceToSequence as STS
from numpy.linalg import norm
import os
import numpy as np
import math
real = np.float64


def Feed_forward_backward_LM(params, in_word_idx_seq, out_word_idx_seq):
    inter_vals = dict()
    grads = dict()
    lower_output_acts = STS.Input_feed_forward(params["W_we"], in_word_idx_seq)
    for i in range(params["num_layers"]):
        layer_name = "LSTM_layer_" + str(i)
        weights = params[layer_name]
        layer_ret = STS.LSTM_feed_forward(weights, lower_output_acts)
        inter_vals[layer_name] = layer_ret
        lower_output_acts = layer_ret[0]
    softmax_ret = STS.Softmax_feed_fordward_backward(params["W_o"],
                                                     lower_output_acts,
                                                     out_word_idx_seq)
    grads["W_o"] = softmax_ret[0]
    sent_log_loss = softmax_ret[2]
    input_errors = softmax_ret[1]
    for i in reversed(range(params["num_layers"])):
        layer_name = "LSTM_layer_" + str(i)
        weights = params[layer_name]
        layer_inter_vals = inter_vals[layer_name]
        layer_ret = STS.LSTM_feed_backward(weights, layer_inter_vals, input_errors)
        grads[layer_name] = layer_ret[0]
        input_errors = layer_ret[1]
    grads["W_we"] = STS.Input_feed_backward(params["W_we"], input_errors,
                                            in_word_idx_seq)
    return grads, sent_log_loss


def Grad_check_LM(params, sample):
    if not os.path.exists("debug"):
        os.mkdir("debug")
    print "start gradient check"
    grads, _ = Feed_forward_backward_LM(params, sample[0], sample[1])
    Check_diff_LM(params, grads, "W_o", sample)
    for i in reversed(range(params["num_layers"])):
        layer_name = "LSTM_layer_" + str(i)
        Check_diff_LM(params, grads, layer_name, sample)
    Check_diff_LM(params, grads, "W_we", sample)
    print "gradient check end"


def Auto_grad_LM(params, fluct_weights, sample):
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
            _, diff1[i, j] = Feed_forward_backward_LM(params, sample[0], sample[1])
            W[i, j] -= 2 * perturbation
            _, diff2[i, j] = Feed_forward_backward_LM(params, sample[0], sample[1])
            # Restore the weight value at (i,j)
            W[i, j] += perturbation
    numerical_grad = (diff1 - diff2) / (2 * perturbation)
    return numerical_grad


def Check_diff_LM(params, grads, name, sample):
    """
    :param params:
    :param grads:
    :param name:
    :param sample:
    :return:
    """
    weights_name = ["W_iota_y", "W_iota_s", "W_phi_y", "W_phi_s", "W",
                    "W_eta_y", "W_eta_s"]
    sigFigs = 5
    if isinstance(params[name], list):
        for i, weights in enumerate(params[name]):
            numDg = Auto_grad_LM(params, weights, sample)
            algDg = grads[name][i]
            diff = algDg - numDg
            minDg = np.minimum(np.abs(algDg), np.abs(numDg))
            minDg[minDg == 0] = 1
            threshold = np.power(10.0, np.maximum(0.0, np.ceil(np.log10(minDg))) - int(sigFigs))
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
        numDg = Auto_grad_LM(params, params[name], sample)
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


def Construct_LM_net(hidden_size_list, we_size, vocab_size, lstm_range=0.08,
                     embedding_range=0.1, softmax_range=0.1):
    """ Mainly for debug
    :param hidden_size_list:
    :param we_size:
    :param vocab_size:
    :param lstm_range:
    :param embedding_range:
    :param softmax_range:
    :return:
    """
    rng = np.random.RandomState(9757)
    # All parameters are saved in a dict
    params = dict()
    params["num_layers"] = len(hidden_size_list)
    # Init in_vocab_word_embedding and out_vocab_word_embedding
    W_we = np.asarray(rng.uniform(low=-embedding_range, high=embedding_range,
                                  size=(we_size, vocab_size)), dtype=real)

    params["W_we"] = W_we
    # Init sentence embedding LSTM weight matrix
    input_size = we_size
    for i, layer_size in enumerate(hidden_size_list):
        layer_name = "LSTM_layer_" + str(i)
        joint_size = layer_size + input_size + 1    # 1 for Bias
        W_iota_y = np.asarray(rng.uniform(low=-lstm_range, high=lstm_range,
                                          size=(layer_size, joint_size)), dtype=real)
        W_iota_s = np.asarray(rng.uniform(low=-lstm_range, high=lstm_range,
                                          size=(layer_size, 1)), dtype=real)
        W_phi_y = np.asarray(rng.uniform(low=-lstm_range, high=lstm_range,
                                         size=(layer_size, joint_size)), dtype=real)
        W_phi_s = np.asarray(rng.uniform(low=-lstm_range, high=lstm_range,
                                         size=(layer_size, 1)), dtype=real)
        W = np.asarray(rng.uniform(low=-lstm_range, high=lstm_range,
                                   size=(layer_size, joint_size)), dtype=real)
        W_eta_y = np.asarray(rng.uniform(low=-lstm_range, high=lstm_range,
                                         size=(layer_size, joint_size)), dtype=real)
        W_eta_s = np.asarray(rng.uniform(low=-lstm_range, high=lstm_range,
                                         size=(layer_size, 1)), dtype=real)
        params[layer_name] = [W_iota_y, W_iota_s, W_phi_y, W_phi_s, W, W_eta_y, W_eta_s]
        input_size = layer_size
    # Init top LSTM to softmax layer weights
    input_size = hidden_size_list[-1]
    layer_size = vocab_size
    joint_size = input_size + 1
    W_o = np.asarray(rng.uniform(low=-softmax_range, high=softmax_range,
                                 size=(layer_size, joint_size)), dtype=real)
    params["W_o"] = W_o
    return params

