# time orientation prev(ious) futu(re)
# layer orientation lower & upper
#            forward <--> backward
#  input activations <--> output errors
# output activations <--> input errors
# embedding phrase & language model phrase
# in means input sequence, out means target sequence which is also used as input
# in language model phrase
# in words are embedding phrase inputs, output words are language model phrase
# inputs
# some code are redundant for clarity

import numpy as np
from numpy.linalg import norm
import math
import time
import os
import cPickle

real = np.float64


# Softmax function
def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    return e / np.sum(e)


# Tanh and derivative
def tanh_prime(x):
    y = np.tanh(x)
    y_prime = 1.0 - (y * y)
    return y, y_prime


# Logistic and derivative
def logistic_prime(x):
    y = 1.0 / (1.0 + np.exp(-x))
    y_prime = y * (1.0 - y)
    return y, y_prime


def linear_prime(x):
    y = x * 1
    y_prime = np.ones(y.shape, dtype=real)
    return y, y_prime


def LSTM_feed_forward(weights, lower_output_acts, init=[]):
    """Feed forward a LSTM layer. This function also contains the feed forward
       between two layers. Specifically a lower layer output activations feed
       forward to this layer
       When this is used to generation, lower_output_acts contains only 1
       time step which is the current step. init contains previous
       configuration. These two things can be used to predict the current
       output.

       :param weights:
               A list of matrix [W_iota_y, W_iota_s, W_phi_y, W_phi_s, W,
                                  W_eta_y, W_eta_s]
               The last column of Weight matrix is bias. So input vectors need
               to append 1.
               input --> gate/cell are weights from the lower layer to this
               layer.
               W_iota_y     input[t] & hidden[t - 1] --> input gate
               W_iota_s                 state[t - 1] --> input gate (peepholes)
               W_phi_y      input[t] & hidden[t - 1] --> forget gate
               W_phi_s                  state[t - 1] --> forget gate
               W            input[t] & hidden[t - 1] --> cell
               W_eta_y      input[t] & hidden[t - 1] --> output gate
               W_eta_s                  state[t - 1] --> output gate
       :param lower_output_acts:
               Output activations of the lower layer.
               It is a matrix. [T, lower_layer_size]
       :param init:
               [hidden output activations, states]
               The initial previous hidden output activations and
               states. Default value is []. The language model is able to
               include the previous hidden states by this arg.
               Both are [layer_size,] vector

       :returns output_acts Y: [time_steps, layer_size] matrix
                states S: [time_steps, layer_size] matrix
                H, Hp:
                G, Gp:
                Y_eta, Yp_eta:
                Y_phi, Yp_phi:
                Y_iota, Yp_iota:
                inputs I: [time_steps, joint_size] matrix
                output_acts is the input to the next layer.
                [output_acts[-1], states[-1]] is the next phrase init
                Others are used for feed backward
    """
    W_iota_y, W_iota_s, W_phi_y, W_phi_s, W, W_eta_y, W_eta_s = weights
    layer_size = W.shape[0]
    time_steps = len(lower_output_acts)
    input_size = lower_output_acts[0].shape[0]
    joint_size = layer_size + input_size + 1  # 1 for bias
    I, X, Y, S = range(time_steps), range(time_steps), range(time_steps), range(time_steps)
    X_iota, Y_iota, Yp_iota = range(time_steps), range(time_steps), range(time_steps)
    X_phi, Y_phi, Yp_phi = range(time_steps), range(time_steps), range(time_steps)
    X_eta, Y_eta, Yp_eta = range(time_steps), range(time_steps), range(time_steps)
    G, Gp, H, Hp = range(time_steps), range(time_steps), range(time_steps), range(time_steps)
    if len(init) == 0:
        prev_hidden = np.zeros((layer_size, 1), dtype=real)
        prev_states = np.zeros((layer_size, 1), dtype=real)
    else:
        prev_hidden, prev_states = init

    for t in range(time_steps):
        # Create the input vector
        I[t] = np.zeros((joint_size, 1), dtype=real)
        if t == 0:
            I[t][0: layer_size] = prev_hidden
            prev_S = prev_states
        else:
            I[t][0: layer_size] = Y[t - 1]
            prev_S = S[t - 1]
        I[t][layer_size: layer_size + input_size] = lower_output_acts[t]
        I[t][layer_size + input_size] = 1   # Bias
        # Calculate input gate activations
        X_iota[t] = W_iota_y.dot(I[t]) + W_iota_s * prev_S
        Y_iota[t], Yp_iota[t] = linear_prime(X_iota[t])
        # Calculate forget gate activations
        X_phi[t] = W_phi_y.dot(I[t]) + W_phi_s * prev_S
        Y_phi[t], Yp_phi[t] = linear_prime(X_phi[t])
        # Calculate cells
        X[t] = W.dot(I[t])
        G[t], Gp[t] = linear_prime(X[t])
        S[t] = Y_phi[t] * prev_S + Y_iota[t] * G[t]
        # Calculate output gate activations
        X_eta[t] = W_eta_y.dot(I[t]) + W_eta_s * S[t]
        Y_eta[t], Yp_eta[t] = linear_prime(X_eta[t])
        # Calculate cell outputs
        H[t], Hp[t] = linear_prime(S[t])
        Y[t] = Y_eta[t] * H[t]
    return Y, S, H, Hp, G, Gp, Y_eta, Yp_eta, Y_phi, Yp_phi, Y_iota, Yp_iota, I


def LSTM_feed_backward(weights, inter_vars, input_errors=0, final=[], prev=[]):
    """Feed backward a LSTM layer. This function also contains the feed backward
       between two layers. Specifically this layer feed back to the lower layer.
       :param weights:
               A list of matrix [W_iota_y, W_iota_s, W_phi_y, W_phi_s, W,
                                  W_eta_y, W_eta_s]
               The last column of Weight matrix is bias. So input vectors need
               to append 1.
               input --> gate/cell are weights form the lower layer to this
               layer.
               W_iota_y     input[t] & hidden[t - 1] --> input gate
               W_iota_s                 state[t - 1] --> input gate (peepholes)
               W_phi_y      input[t] & hidden[t - 1] --> forget gate
               W_phi_s                  state[t - 1] --> forget gate
               W            input[t] & hidden[t - 1] --> cell
               W_eta_y      input[t] & hidden[t - 1] --> output gate
               W_eta_s                  state[t - 1] --> output gate
       :param inter_vars:
               [Y, S, H, Hp, G, Gp, Y_eta, Yp_eta, Y_phi, Yp_phi,
                Y_iota, Yp_iota, I]
               The intermediate variables calculated in the feed forward phrase.
               They are used to calculate the derivative of the weights
       :param input_errors: [time_steps, layer_size] matrix
              The error come from the upper layer. Note that they have
              multiplied by the upper weight matrix transpose. So they are just
              the errors injected to this layer.
       :param final: for idx t + 1. 5 terms
              [D_hidden_units, D_states, D_forget_gate, D_input_gate, Y_phi_futu]
              The final future hidden unites, states, forget gate, input gate
              derivatives. Default value is []. The embedding phrase is able to
              receive errors from the language model phrase.
              4 terms because:
              S[t - 1] --> input gate[t]
              S[t - 1] --> forget_gate[t]
              S[t - 1] --> S[t]
              H[t - 1] --> Cell[t]
       :param prev: for idx t - 1. 1 term
              the States values before 0 time for lm LSTM
       :return Dg_iota_y: Corresponding to the weights
               Dg_iota_s:
               Dg_phi_y:
               Dg_phi_s:
               Dg:
               Dg_eta_y:
               Dg_eta_s:
               lower_input_error: error feed back from input gate, forget gate,
               cell, output gate.
               prev_final: used to pass training variable from language model
               phrase to embedding phrase
    """
    W_iota_y, W_iota_s, W_phi_y, W_phi_s, W, W_eta_y, W_eta_s = weights
    Y, S, H, Hp, G, Gp, Y_eta, Yp_eta, Y_phi, Yp_phi, Y_iota, Yp_iota, I = inter_vars
    layer_size = W.shape[0]
    time_steps = len(I)
    joint_szie = I[0].shape[0]
    input_size = joint_szie - layer_size - 1    # 1 for bias
    # Initialize gradient vectors
    lower_input_errors = range(time_steps)
    Dg = np.zeros((layer_size, joint_szie), dtype=real)
    Dg_eta_y = np.zeros((layer_size, joint_szie), dtype=real)
    Dg_eta_s = np.zeros((layer_size, 1), dtype=real)
    Dg_phi_y = np.zeros((layer_size, joint_szie), dtype=real)
    Dg_phi_s = np.zeros((layer_size, 1), dtype=real)
    Dg_iota_y = np.zeros((layer_size, joint_szie), dtype=real)
    Dg_iota_s = np.zeros((layer_size, 1), dtype=real)
    # Save the last deltas necessary
    if len(final) == 0:
        Dl_futu = np.zeros((layer_size, 1), dtype=real)
        dE_futu = np.zeros((layer_size, 1), dtype=real)
        Dl_eta_futu = np.zeros((layer_size, 1), dtype=real)
        Dl_phi_futu = np.zeros((layer_size, 1), dtype=real)
        Dl_iota_futu = np.zeros((layer_size, 1), dtype=real)
        Y_phi_futu = np.zeros((layer_size, 1), dtype=real)
    else:
        # Embedding net receive error from lm net
        Dl_futu, dE_futu, Dl_eta_futu, Dl_phi_futu, Dl_iota_futu, Y_phi_futu = final
    if len(prev) == 0:
        prev_states = np.zeros((layer_size, 1), dtype=real)
    else:
        prev_states = prev[0]

    # Calculate the error and add it
    for t in reversed(range(time_steps)):
        if t == 0:
            prev_S = prev_states
        else:
            prev_S = S[t - 1]
        # Calculate the epsilon
        if input_errors == 0:
            # the top layer of embedding net receives no errors form upper layer
            Eps = (W.T[0: layer_size].dot(Dl_futu) +
                   W_eta_y.T[0: layer_size].dot(Dl_eta_futu) +
                   W_phi_y.T[0: layer_size].dot(Dl_phi_futu) +
                   W_iota_y.T[0: layer_size].dot(Dl_iota_futu))
        else:
            Eps = (input_errors[t] +
                   W.T[0: layer_size].dot(Dl_futu) +
                   W_eta_y.T[0: layer_size].dot(Dl_eta_futu) +
                   W_phi_y.T[0: layer_size].dot(Dl_phi_futu) +
                   W_iota_y.T[0: layer_size].dot(Dl_iota_futu))
        # Calculate the change in output gates
        Dl_eta = Yp_eta[t] * Eps * H[t]    # element wise multiplication
        Dg_eta_y += Dl_eta.dot(I[t].T)
        Dg_eta_s += Dl_eta * S[t]
        # Calculate the derivative of the error feed back to states
        dE = (Eps * Y_eta[t] * Hp[t] +
              dE_futu * Y_phi_futu +
              Dl_iota_futu * W_iota_s +
              Dl_phi_futu * W_phi_s +
              Dl_eta * W_eta_s)
        # Calculate the delta of the states
        Dl = Y_iota[t] * Gp[t] * dE
        Dg += Dl.dot(I[t].T)
        # Calculate the delta of forget gate
        Dl_phi = Yp_phi[t] * dE * prev_S
        Dg_phi_y += Dl_phi.dot(I[t].T)
        Dg_phi_s += Dl_phi * prev_S
        # Calculate the delta of input gate
        Dl_iota = Yp_iota[t] * dE * G[t]
        Dg_iota_y += Dl_iota.dot(I[t].T)
        Dg_iota_s += Dl_iota * prev_S
        # Save the future ones
        Dl_futu = Dl
        dE_futu = dE
        Dl_eta_futu = Dl_eta
        Dl_phi_futu = Dl_phi
        Dl_iota_futu = Dl_iota
        Y_phi_futu = Y_phi[t]
        lower_input_errors[t] = (W_iota_y.T[layer_size: layer_size + input_size].dot(Dl_iota) +
                                 W_phi_y.T[layer_size: layer_size + input_size].dot(Dl_phi) +
                                 W.T[layer_size: layer_size + input_size].dot(Dl) +
                                 W_eta_y.T[layer_size: layer_size + input_size].dot(Dl_eta))
    return ([Dg_iota_y, Dg_iota_s, Dg_phi_y, Dg_phi_s, Dg, Dg_eta_y, Dg_eta_s],
            lower_input_errors,
            [Dl_futu, dE_futu, Dl_eta_futu, Dl_phi_futu, Dl_iota_futu, Y_phi_futu])


# Direct input version, more time consuming in matrix multiplication
# Modify LSTM_feed_forward, this can be avoided. But in such condition, the
# feed_forward function of the first LSTM layer is inconsistent with other
# layers. Add a switch to indicate first layer LSTM.
# Word-embedding version
def Input_feed_forward(W_we, word_idx_seq):
    """Transfer word_idx_seq to word_embedding_seq. The weights from
    :param W_we
    :param word_idx_seq:
    :return word_embedding_seq:
    """
    time_steps = len(word_idx_seq)
    we_seq = range(time_steps)
    for t in range(time_steps):
        we_seq[t] = np.asarray([W_we.T[word_idx_seq[t]]]).T
    return we_seq


def Input_feed_backward(W_we, input_errors, word_idx_seq):
    """

    :param W_we:
    :param input_errors:
    :param word_idx_seq:
    :return:
    """
    time_steps = len(input_errors)
    Dg_we = np.zeros(W_we.shape, dtype=real)
    for t in reversed(range(time_steps)):
        word_idx = word_idx_seq[t]
        Dg_we[:, word_idx] += input_errors[t][:, 0]
    return Dg_we


def Softmax_feed_fordward_backward(W_o, lower_output_acts, target_idx_seq):
    """
    :param W_o:
    :param lower_output_acts:
    :param target_idx_seq:
    :return Dg_o
            lower_input_errors
            sent_log_loss
    """
    time_steps = len(lower_output_acts)
    layer_size = W_o.shape[0]
    joint_size = W_o.shape[1]
    input_size = W_o.shape[1] - 1   # 1 for Bias
    # Softmax feed forward
    sent_log_loss = 0
    Y, X_o, Y_o = range(time_steps), range(time_steps), range(time_steps)
    for t in range(time_steps):
        # Calculate the emission
        Y[t] = np.zeros((joint_size, 1), dtype=real)
        Y[t][:input_size] = lower_output_acts[t]
        Y[t][-1] = 1    # 1 for Bias
        X_o[t] = W_o.dot(Y[t])
        Y_o[t] = softmax(X_o[t])
        # sent_log_loss = - sent_log_probability
        sent_log_loss -= math.log(max(Y_o[t][target_idx_seq[t]], 1e-20))
    # Softmax feed backward
    lower_input_errors = range(time_steps)
    Dg_o = np.zeros((layer_size, joint_size), dtype=real)
    for t in reversed(range(time_steps)):
        # A little different from Neubig's code
        Dl_o = Y_o[t] * 1
        Dl_o[target_idx_seq[t]] -= 1
        Dg_o += Dl_o.dot(Y[t].T)
        lower_input_errors[t] = W_o.T[0: input_size].dot(Dl_o)
    return Dg_o, lower_input_errors, sent_log_loss


def Construct_net(hidden_size_list, we_size, in_vocab_size, out_vocab_size=0,
                  lstm_range=0.08, embedding_range=0.1, softmax_range=0.1):
    """This version must contain a word_embedding layer
    :rtype : dict
    :param hidden_size_list: one value for each LSTM layer
    :param we_size: the word_embedding layer size.
    :param in_vocab_size: used to define the word_embedding matrix
    :param out_vocab_size: default is 0. If it is 0, the input vocab and output
           vocab are the same vocab, and the word_embedding matrix is also the
           same
    :param lstm_range: initial weight range
    :param softmax_range:
    :param embedding_range:
    :return A dict of weight matrix which defines the network
    """
    rng = np.random.RandomState(89757)
    # All parameters are saved in a dict
    params = dict()
    params["num_layers"] = len(hidden_size_list)
    # Init in_vocab_word_embedding and out_vocab_word_embedding
    W_we_in = np.asarray(rng.uniform(low=-embedding_range, high=embedding_range,
                                     size=(we_size, in_vocab_size)), dtype=real)
    if out_vocab_size == 0:
        W_we_out = W_we_in
        out_vocab_size = in_vocab_size
    else:
        W_we_out = np.asarray(rng.uniform(low=-embedding_range, high=embedding_range,
                                          size=(we_size, out_vocab_size)), dtype=real)
    params["W_we_in"] = W_we_in
    params["W_we_out"] = W_we_out
    # Init sentence embedding LSTM weight matrix
    input_size = we_size
    for i, layer_size in enumerate(hidden_size_list):
        layer_name = "em_LSTM_layer_" + str(i)
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
    # Init language model LSTM weight matrix
    input_size = we_size
    for i, layer_size in enumerate(hidden_size_list):
        layer_name = "lm_LSTM_layer_" + str(i)
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
    layer_size = out_vocab_size
    joint_size = input_size + 1
    W_o = np.asarray(rng.uniform(low=-softmax_range, high=softmax_range,
                                 size=(layer_size, joint_size)), dtype=real)
    params["W_o"] = W_o
    return params


def Feed_forward_backward(params, in_word_idx_seq, out_word_idx_seq,
                          target_word_idx_seq):
    """
    :param params:
    :param in_word_idx_seq:
    :param out_word_idx_seq: start form <EOS>, end with last word
    :param target_word_idx_seq: start form first word, end with <EOS>
    :return a dict of grads
            sent_log_loss
    """
    inter_vals = dict()
    grads = dict()
    # Word embedding layers feed forward
    em_lower_output_acts = Input_feed_forward(params["W_we_in"], in_word_idx_seq)
    lm_lower_output_acts = Input_feed_forward(params["W_we_out"], out_word_idx_seq)
    # LSTM layers feed forward
    for i in range(params["num_layers"]):
        em_layer_name = "em_LSTM_layer_" + str(i)
        lm_layer_name = "lm_LSTM_layer_" + str(i)
        em_weights = params[em_layer_name]
        lm_weights = params[lm_layer_name]
        # Feed forward embedding layer
        em_layer_ret = LSTM_feed_forward(em_weights, em_lower_output_acts)
        # Get lm initial hidden activations and states
        em_output_acts = em_layer_ret[0]
        em_states = em_layer_ret[1]
        lm_init = [em_output_acts[-1], em_states[-1]]
        # Feed forward lm layer, start form init activations and states
        lm_layer_ret = LSTM_feed_forward(lm_weights, lm_lower_output_acts, lm_init)
        # Save the intermediate values for feed backward
        inter_vals[em_layer_name] = em_layer_ret
        inter_vals[lm_layer_name] = lm_layer_ret
        # Loop variant assignment
        em_lower_output_acts = em_layer_ret[0]
        lm_lower_output_acts = lm_layer_ret[0]
    # Softmax layers feed forward and backward
    softmax_lower_output_acts = lm_lower_output_acts
    softmax_ret = Softmax_feed_fordward_backward(params["W_o"],
                                                 softmax_lower_output_acts, target_word_idx_seq)
    grads["W_o"] = softmax_ret[0]
    sent_log_loss = softmax_ret[2]
    # Feed back LSTM layers
    lm_input_errors = softmax_ret[1]
    em_input_errors = 0
    for i in reversed(range(params["num_layers"])):
        em_layer_name = "em_LSTM_layer_" + str(i)
        lm_layer_name = "lm_LSTM_layer_" + str(i)
        em_weights = params[em_layer_name]
        lm_weights = params[lm_layer_name]
        em_inter_vals = inter_vals[em_layer_name]
        lm_inter_vals = inter_vals[lm_layer_name]
        # Feed back language layer
        prev = [em_inter_vals[1][-1]]
        lm_layer_ret = LSTM_feed_backward(lm_weights, lm_inter_vals,
                                          lm_input_errors, prev=prev)
        # Get the future error for embedding layer
        em_final = lm_layer_ret[2]
        # Feed back em layer
        em_layer_ret = LSTM_feed_backward(em_weights, em_inter_vals,
                                          em_input_errors, final=em_final)
        # Save the gradients
        grads[em_layer_name] = em_layer_ret[0]
        grads[lm_layer_name] = lm_layer_ret[0]
        # Loop variable assignment
        em_input_errors = em_layer_ret[1]
        lm_input_errors = lm_layer_ret[1]
    # Feed back in and out word embedding
    in_we_errors = em_input_errors
    out_we_errors = lm_input_errors
    grads["W_we_in"] = Input_feed_backward(params["W_we_in"],
                                           in_we_errors, in_word_idx_seq)
    grads["W_we_out"] = Input_feed_backward(params["W_we_out"],
                                            out_we_errors, out_word_idx_seq)

    return grads, sent_log_loss


# learn_rate strategy exists in a outer loop
# error clip strategy is include in this function
def Weight_SGD(weights, gradients, learn_rate=0.1, clip_norm=0):
    """Change the one weight matrix. Not all the matrix
    :param weights:
    :param gradients:
    :param learn_rate:
    :param clip_norm: if clip_norm = 0, do not clip gradients
    :return None
    """
    if clip_norm > 0:
        grads_norm = gradients * gradients
        clip_idx = grads_norm > clip_norm
        gradients[clip_idx] = clip_norm * gradients[clip_idx] / grads_norm[clip_idx]
    weights -= learn_rate * gradients


def All_params_SGD(params, grads, ff_learn_rate=0.7, lstm_learn_rate=0.7,
                   lstm_clip_norm=5):
    """
    :param params:
    :param grads:
    :param ff_learn_rate: learning rate for word_embedding and softmax weights
    :param lstm_learn_rate:
    :param lstm_clip_norm:
    :return:
    """
    Weight_SGD(params["W_o"], grads["W_o"], learn_rate=ff_learn_rate)
    Weight_SGD(params["W_we_in"], grads["W_we_in"], learn_rate=ff_learn_rate)
    Weight_SGD(params["W_we_out"], grads["W_we_out"], learn_rate=ff_learn_rate)
    for i in range(params["num_layers"]):
        em_layer_name = "em_LSTM_layer_" + str(i)
        lm_layer_name = "lm_LSTM_layer_" + str(i)
        for em_weights, em_gradients in zip(params[em_layer_name], grads[em_layer_name]):
            Weight_SGD(em_weights, em_gradients, learn_rate=lstm_learn_rate,
                       clip_norm=lstm_clip_norm)
        for lm_weights, lm_gradients in zip(params[lm_layer_name], grads[lm_layer_name]):
            Weight_SGD(lm_weights, lm_gradients, learn_rate=lstm_learn_rate,
                       clip_norm=lstm_clip_norm)


# Did not use minibatch
def Train(params, samples, epochs = 8, lr_decay=True):
    """
    :param params:
    :param samples: every sample is a list [in_word_idx_seq, outword_word_idx_seq,
           target_word_idx_seq], every word_idx_seq is a list of int. All samples
           are contained in a list
    :param epochs:
    :param lr_decay: learn rate decay
    :return:
    """
    trained_epochs = 0
    ff_lr = 0.1
    lstm_lr = 0.7
    while True:
        trained_samples = 0
        log_loss = 0
        trained_words = 0
        t0 = time.time()
        # If you want to do parallel computation. Train several samples and
        # accumulate the grads
        for sample in samples:
            # Train the net
            grads, sent_ll = Feed_forward_backward(params, sample[0], sample[1],
                                                   sample[2])
            All_params_SGD(params, grads, ff_learn_rate=ff_lr,
                           lstm_learn_rate=lstm_lr, lstm_clip_norm=5)
            log_loss += sent_ll
            trained_words += len(sample[0]) + len(sample[1])
            # Out put some info
            trained_samples += 1
            if trained_samples % 100 == 0:
                t1 = time.time()
                print "Average sentence log loss: %.5f" % log_loss / 1000.0,
                print "\tword log loss: %.5f" % log_loss / float(trained_words),
                print "\twords/s: %dk" % trained_words / 1000.0 / float(t1 - t0)
                log_loss = 0
                trained_words = 0
                t0 = t1
            if trained_epochs >= 5:
                # half samples clip
                if trained_samples == len(samples) / 2:
                    ff_lr /= 2
                    lstm_lr /= 2
                if trained_samples == len(samples):
                    ff_lr /= 2
                    lstm_lr /= 2
        trained_epochs += 1


def Grad_check(params, sample):
    """
    :param params:
    :param sample:
    :return:
    """
    # Create a folder to save the failed gradient check
    if not os.path.exists("debug"):
        os.mkdir("debug")
    print "start gradient check"
    grads, _ = Feed_forward_backward(params, sample[0], sample[1], sample[2])
    Check_diff(params, grads, "W_o", sample)
    for i in reversed(range(params["num_layers"])):
        em_layer_name = "em_LSTM_layer_" + str(i)
        lm_layer_name = "lm_LSTM_layer_" + str(i)
        Check_diff(params, grads, em_layer_name, sample)
        Check_diff(params, grads, lm_layer_name, sample)
    Check_diff(params, grads, "W_we_in", sample)
    Check_diff(params, grads, "W_we_out", sample)
    print "gradient check end"



def Auto_grad(params, fluct_weights, sample):
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
            _, diff1[i, j] = Feed_forward_backward(params, sample[0], sample[1],
                                                   sample[2])
            W[i, j] -= 2 * perturbation
            _, diff2[i, j] = Feed_forward_backward(params, sample[0], sample[1],
                                                   sample[2])
            # Restore the weight value at (i,j)
            W[i, j] += perturbation
    numerical_grad = (diff1 - diff2) / (2 * perturbation)
    return numerical_grad





def Check_diff(params, grads, name, sample):
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
            numDg = Auto_grad(params, weights, sample)
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
        numDg = Auto_grad(params, params[name], sample)
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


def Save_params(params, name="STS.pkl"):
    """Save parameters
    :param params:
    :param name:
    :return:
    """
    pkl_file = open(name, "wb")
    cPickle.dump(params, pkl_file)
    pkl_file.close()


def Load_params(name="STS.pkl"):
    """Load dict of parameters
    :param name:
    :return:
    """
    pkl_file = open(name, "rb")
    return cPickle.load(pkl_file)


def Generate(params, in_word_idx_seq):
    """Generate a sequence
    :param params:
    :param in_word_idx_seq:
    :return:
    """
    None
