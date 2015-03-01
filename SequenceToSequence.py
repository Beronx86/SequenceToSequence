# time orientation prev(ious) futu(re)
# layer orientation lower & upper
#            forward <--> backward
#  input activations <--> output errors
# output activations <--> input errors
# embedding phase & language model phase
# in means input sequence, out means target sequence which is also used as input
# in language model phase
# in words are embedding phase inputs, output words are language model phase
# inputs
# some code are redundant for clarity
# 0 <NUL>  1 <EOS>  2 <UNK>

import numpy as np
from numpy.linalg import norm
import math
import time
import os
import cPickle

real = np.float32


# Softmax function
def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    return e / np.sum(e, axis=0)


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


def LSTM_feed_forward(weights, lower_output_acts, init=[], in_seq_lens=[]):
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
               states. Default value is []. In sequence to sequence learning,
               the language model is able to include the previous hidden states
               through this arg. Note that when these values propagate from EM
               to LM, they are multiplied by LM recurrent weights
               Both are [layer_size,] vector
       :param in_seq_lens:
               Rectify hidden activations for different length sequence

       :returns output_acts Y: [time_steps, layer_size] matrix
                states S: [time_steps, layer_size] matrix
                H, Hp:
                G, Gp:
                Y_eta, Yp_eta:
                Y_phi, Yp_phi:
                Y_iota, Yp_iota:
                inputs I: [time_steps, joint_size] matrix
                output_acts is the input to the upper layer.
                [output_acts[-1], states[-1]] is the next phase init
                Others are used for feed backward
    """
    W_iota_y, W_iota_s, W_phi_y, W_phi_s, W, W_eta_y, W_eta_s = weights
    layer_size = W.shape[0]
    time_steps = len(lower_output_acts)
    input_size = lower_output_acts[0].shape[0]
    joint_size = layer_size + input_size + 1  # 1 for bias
    batch_size = lower_output_acts[-1].shape[1]
    I, X, Y, S = range(time_steps), range(time_steps), range(time_steps), range(time_steps)
    X_iota, Y_iota, Yp_iota = range(time_steps), range(time_steps), range(time_steps)
    X_phi, Y_phi, Yp_phi = range(time_steps), range(time_steps), range(time_steps)
    X_eta, Y_eta, Yp_eta = range(time_steps), range(time_steps), range(time_steps)
    G, Gp, H, Hp = range(time_steps), range(time_steps), range(time_steps), range(time_steps)
    seq_lens = np.asarray(in_seq_lens)

    prev_hidden = 0
    prev_states = 0
    if len(init) != 0:
        prev_hidden, prev_states = init

    for t in range(time_steps):
        # Create the input vector
        if seq_lens.shape[0] == 0:
            not_start = 0
        else:
            not_start = (t < time_steps - seq_lens)

        I[t] = np.zeros((joint_size, batch_size), dtype=real)
        if t == 0:
            I[t][0: layer_size] = prev_hidden
            prev_S = prev_states
        else:
            I[t][0: layer_size] = Y[t - 1]
            prev_S = S[t - 1]
        if np.sum(not_start) == 0:
            I[t][layer_size: layer_size + input_size] = lower_output_acts[t]
            I[t][layer_size + input_size] = 1   # Bias
        else:
            start = np.invert(not_start)
            I[t][layer_size: layer_size + input_size, start] = lower_output_acts[t][:, start]
            I[t][layer_size + input_size, start] = 1

        # Calculate input gate activations
        X_iota[t] = W_iota_y.dot(I[t]) + W_iota_s * prev_S
        Y_iota[t], Yp_iota[t] = logistic_prime(X_iota[t])
        # Calculate forget gate activations
        X_phi[t] = W_phi_y.dot(I[t]) + W_phi_s * prev_S
        Y_phi[t], Yp_phi[t] = logistic_prime(X_phi[t])
        # Calculate cells
        X[t] = W.dot(I[t])
        G[t], Gp[t] = tanh_prime(X[t])
        S[t] = Y_phi[t] * prev_S + Y_iota[t] * G[t]
        # Calculate output gate activations
        X_eta[t] = W_eta_y.dot(I[t]) + W_eta_s * S[t]
        Y_eta[t], Yp_eta[t] = logistic_prime(X_eta[t])
        # Calculate cell outputs
        H[t], Hp[t] = tanh_prime(S[t])
        Y[t] = Y_eta[t] * H[t]
        # Rectify the activations for not start sequences.
        if np.sum(not_start) > 0:
            Y_iota[t][:, not_start] = 0
            Yp_iota[t][:, not_start] = 0
            Y_phi[t][:, not_start] = 0
            Yp_phi[t][:, not_start] = 0
            # G[t][:, not_start] = 0
            Gp[t][:, not_start] = 0
            # S[t][:, not_start] = 0
            Y_eta[t][:, not_start] = 0
            Yp_eta[t][:, not_start] = 0
            # H[t][:, not_start] = 0
            Hp[t][:, not_start] = 0
            # Y[t][:, not_start] = 0

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
               The intermediate variables calculated in the feed forward phase.
               They are used to calculate the derivative of the weights
       :param input_errors: [time_steps, layer_size] matrix
              The error come from the upper layer. Note that they have
              multiplied by the upper weight matrix transpose. So they are just
              the errors injected to this layer.
       :param final: errors for idx t + 1.
              The final future hidden unites, states, forget gate, input gate
              derivatives. Default value is []. The embedding phase is able to
              receive errors from the language model phase.
              Specifically these are the errors form LM phase 0 time to EM phase
              T-1 time. Note that values form EM to LM are multiplied by LM
              recurrent weights. In error back propagation, errors should also
              be multiplied by these weights.
              [Dl_recur_errors, Dl_eta_recur_errors, Dl_phi_recur_errors,
               Dl_iota_recur_errors, dE_recur_errors, dE_phi_errors,
               dE_iota_errors]
              Dl_recur_errors       errors from t + 1 states to t cell
              Dl_eta_recur_errors   errors form t + 1 output gate to t cell
              Dl_phi_recur_errors   errors form t + 1 forget gate to t cell
              Dl_iota_recur_errors  errors from t + 1 input gate to t cell
              dE_recur_errors    errors form t + 1 to t states
              dE_phi_errors      errors from t + 1 forget gate to t states (peepholes)
              dE_iota_errors     errors from t + 1 input gate to t states (peepholes)
       :param prev: for idx t - 1. 1 term
              the States values before 0 time for LM LSTM
       :returns gradient_list: the orders are corresponding to weights order
                lower_input_errors: the errors injected to lower layer. They are
                fed back states, output gate, forget gate, input gate
                recur_errors: errors form LM phase to EM phase
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
    Dl_recur_errors = 0
    Dl_eta_recur_errors = 0
    Dl_phi_recur_errors = 0
    Dl_iota_recur_errors = 0
    dE_recur_errors = 0
    dE_phi_errors = 0
    dE_iota_errors = 0
    prev_states = 0
    if len(final) != 0:
        # Embedding net receive error from lm net
        (Dl_recur_errors, Dl_eta_recur_errors, Dl_phi_recur_errors,
         Dl_iota_recur_errors, dE_recur_errors, dE_phi_errors, dE_iota_errors) = final
    if len(prev) != 0:
        prev_states = prev[0]

    # Calculate the error and add it
    for t in reversed(range(time_steps)):
        # Calculate the epsilon
        if input_errors == 0:
            # the top layer of embedding net receives no errors form upper
            # layer
            Eps = (Dl_recur_errors +
                   Dl_eta_recur_errors +
                   Dl_phi_recur_errors +
                   Dl_iota_recur_errors)
        else:
            Eps = (input_errors[t] +
                   Dl_recur_errors +
                   Dl_eta_recur_errors +
                   Dl_phi_recur_errors +
                   Dl_iota_recur_errors)
        if t == 0:
            prev_S = prev_states
        else:
            prev_S = S[t - 1]
        # Calculate the change in output gates
        Dl_eta = Yp_eta[t] * Eps * H[t]    # element wise multiplication
        Dg_eta_y += Dl_eta.dot(I[t].T)
        Dg_eta_s += np.sum(Dl_eta * S[t], axis=1).reshape(layer_size, 1)
        # Calculate the derivative of the error feed back to states
        dE = (Eps * Y_eta[t] * Hp[t] +  # states to cell output weights
              Dl_eta * W_eta_s +        # peepholes to forget gate
              dE_recur_errors +         # s[t - 1] to s[t] weights
              dE_phi_errors +           # peepholes to forget gate
              dE_iota_errors)           # peepholes to input gate
        # Calculate the delta of the states
        Dl = Y_iota[t] * Gp[t] * dE
        Dg += Dl.dot(I[t].T)
        # Calculate the delta of forget gate
        Dl_phi = Yp_phi[t] * dE * prev_S
        Dg_phi_y += Dl_phi.dot(I[t].T)
        Dg_phi_s += np.sum(Dl_phi * prev_S, axis=1).reshape(layer_size, 1)
        # Calculate the delta of input gate
        Dl_iota = Yp_iota[t] * dE * G[t]
        Dg_iota_y += Dl_iota.dot(I[t].T)
        Dg_iota_s += np.sum(Dl_iota * prev_S, axis=1).reshape(layer_size, 1)
        # The errors inject to the previous time step
        Dl_recur_errors = W.T[0: layer_size].dot(Dl)
        Dl_eta_recur_errors = W_eta_y.T[0: layer_size].dot(Dl_eta)
        Dl_phi_recur_errors = W_phi_y.T[0: layer_size].dot(Dl_phi)
        Dl_iota_recur_errors = W_iota_y.T[0: layer_size].dot(Dl_iota)
        # The states recurrent errors
        dE_recur_errors = dE * Y_phi[t]
        dE_phi_errors = Dl_phi * W_phi_s
        dE_iota_errors = Dl_iota * W_iota_s
        # The errors inject to lower layer
        lower_input_errors[t] = (W_iota_y.T[layer_size: layer_size + input_size].dot(Dl_iota) +
                                 W_phi_y.T[layer_size: layer_size + input_size].dot(Dl_phi) +
                                 W.T[layer_size: layer_size + input_size].dot(Dl) +
                                 W_eta_y.T[layer_size: layer_size + input_size].dot(Dl_eta))
        # The weights between EM and LM are the LM recurrent weights. So when
        # calculate the errors form LM to EM, the LM recurrent weights are used.
    return ([Dg_iota_y, Dg_iota_s, Dg_phi_y, Dg_phi_s, Dg, Dg_eta_y, Dg_eta_s],
            lower_input_errors,
            [Dl_recur_errors, Dl_eta_recur_errors, Dl_phi_recur_errors,
             Dl_iota_recur_errors, dE_recur_errors, dE_phi_errors, dE_iota_errors])


# Direct input version, more time consuming in matrix multiplication
# Modify LSTM_feed_forward, this can be avoided. But in such condition, the
# feed_forward function of the first LSTM layer is inconsistent with other
# layers. Add a switch to indicate first layer LSTM.
# Word-embedding version
def Input_feed_forward(W_we, word_idx_seq):
    """Transfer word_idx_seq to word_embedding_seq. The weights from
    :param W_we
    :param word_idx_seq: A sample mini-batch. It is a list of lists.
           The outer list is of length T and indexed by time. The inner list is
           of lengths K and indexed by different samples. All the samples are
           made to the same length.
    :return word_embedding_seq:
    """
    time_steps = len(word_idx_seq)
    we_seq = range(time_steps)
    for t in range(time_steps):
        we_seq[t] = W_we[:, word_idx_seq[t]]    # Every column is a sample
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
        # Dg_we[:, word_idx] += input_errors[t]
        # If different samples include the same word at the same time, the above
        # line is wrong.
        for i, wi in enumerate(word_idx):
            Dg_we[:, wi] += input_errors[t][:, i]
    return Dg_we


def Softmax_feed_fordward(W_o, lower_output_acts):
    """
    :param W_o:
    :param lower_output_acts:
    :return Dg_o
            lower_input_errors
            sent_log_loss
    """
    time_steps = len(lower_output_acts)
    layer_size = W_o.shape[0]
    joint_size = W_o.shape[1]
    input_size = W_o.shape[1] - 1   # 1 for Bias
    batch_size = lower_output_acts[0].shape[1]
    # Softmax feed forward
    Y, X_o, Y_o = range(time_steps), range(time_steps), range(time_steps)
    for t in range(time_steps):
        # Calculate the emission
        Y[t] = np.zeros((joint_size, batch_size), dtype=real)
        Y[t][:input_size] = lower_output_acts[t]
        Y[t][-1] = 1    # 1 for Bias
        X_o[t] = W_o.dot(Y[t])
        Y_o[t] = softmax(X_o[t])
    return Y, Y_o


def Softmax_feed_backward(W_o, inter_vals, target_idx_seq, out_seq_lens):
    """
    :param W_o:
    :param inter_vals:
    :param target_idx_seq: It is a list of lists and has the same shape as
           word_idx_seq
    :param out_seq_lens: Record every sample length in the mini-batch. It the length
           of a sample is smaller than that of the mini-batch, the gradients of
           the excessive steps are all set to 0. So they have no impact on
           training
    :return:
    """
    # Softmax feed backward
    Y, Y_o = inter_vals
    time_steps = len(target_idx_seq)
    layer_size = W_o.shape[0]
    joint_size = W_o.shape[1]
    input_size = joint_size - 1
    batch_size = Y[0].shape[1]
    seq_lens = np.asarray(out_seq_lens)
    lower_input_errors = range(time_steps)
    Dg_o = np.zeros((layer_size, joint_size), dtype=real)
    batch_log_loss = 0
    for t in reversed(range(time_steps)):
        # A little different from Neubig's code
        Dl_o = Y_o[t] * 1
        Dl_o[target_idx_seq[t], range(batch_size)] -= 1
        # If t is larger than the length of a sample, this sample already end.
        end = (t >= seq_lens)
        Dl_o[:, end] = 0
        Dg_o += Dl_o.dot(Y[t].T)
        lower_input_errors[t] = W_o.T[0: input_size].dot(Dl_o)
        # Exclude the log values of the excessive time steps.
        Y_o[t][:, end] = 1
        target_prob = np.maximum(Y_o[t][target_idx_seq[t], range(batch_size)],
                                 1e-20)
        batch_log_loss -= np.sum(np.log(target_prob))
    return Dg_o, lower_input_errors, batch_log_loss


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
    rng = np.random.RandomState()
    # All parameters are saved in a dict
    params = dict()
    params["num_layers"] = len(hidden_size_list)
    # Init in_vocab_word_embedding and out_vocab_word_embedding
    W_we_in = np.asarray(rng.uniform(low=-embedding_range, high=embedding_range,
                                     size=(we_size, in_vocab_size)), dtype=real)
    W_we_in[:, 0] = 0   # Column 0 is <NUL>
    if out_vocab_size == 0:
        W_we_out = W_we_in
        out_vocab_size = in_vocab_size
    else:
        W_we_out = np.asarray(rng.uniform(low=-embedding_range, high=embedding_range,
                                          size=(we_size, out_vocab_size)), dtype=real)
        W_we_out[:, 0] = 0  # Column 0 is <NUL>
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
                          target_word_idx_seq, in_seq_lens, out_seq_lens):
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
        em_layer_ret = LSTM_feed_forward(em_weights, em_lower_output_acts,
                                         in_seq_lens=in_seq_lens)
        # Get lm initial hidden activations and states
        em_output_acts = em_layer_ret[0]
        em_states = em_layer_ret[1]
        lm_init = [em_output_acts[-1], em_states[-1]]
        # Feed forward lm layer, start form init activations and states
        lm_layer_ret = LSTM_feed_forward(lm_weights, lm_lower_output_acts,
                                         lm_init)
        # Save the intermediate values for feed backward
        inter_vals[em_layer_name] = em_layer_ret
        inter_vals[lm_layer_name] = lm_layer_ret
        # Loop variant assignment
        em_lower_output_acts = em_layer_ret[0]
        lm_lower_output_acts = lm_layer_ret[0]
    # Softmax layers feed forward and backward
    softmax_lower_output_acts = lm_lower_output_acts
    softmax_inter_values = Softmax_feed_fordward(params["W_o"],
                                                 softmax_lower_output_acts)
    softmax_ret = Softmax_feed_backward(params["W_o"], softmax_inter_values,
                                        target_word_idx_seq, out_seq_lens)
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
def Weight_SGD(weights, gradients, batch_size, learn_rate=0.1, clip_norm=0):
    """Change the one weight matrix. Not all the matrix
    :param weights:
    :param gradients:
    :param learn_rate:
    :param clip_norm: if clip_norm = 0, do not clip gradients
    :return None
    """
    if clip_norm > 0:
        grads_norm = gradients * gradients
        clip_idx = grads_norm > clip_norm * clip_norm
        gradients[clip_idx] = clip_norm * gradients[clip_idx] / grads_norm[clip_idx]
    weights -= learn_rate * gradients / batch_size


def All_params_SGD(params, grads, batch_size, ff_learn_rate=0.7,
                   lstm_learn_rate=0.7, lstm_clip_norm=5):
    """
    :param params:
    :param grads:
    :param ff_learn_rate: learning rate for word_embedding and softmax weights
    :param lstm_learn_rate:
    :param lstm_clip_norm:
    :return:
    """
    Weight_SGD(params["W_o"], grads["W_o"], batch_size, learn_rate=ff_learn_rate)
    Weight_SGD(params["W_we_in"], grads["W_we_in"], batch_size, learn_rate=ff_learn_rate)
    Weight_SGD(params["W_we_out"], grads["W_we_out"], batch_size, learn_rate=ff_learn_rate)
    for i in range(params["num_layers"]):
        em_layer_name = "em_LSTM_layer_" + str(i)
        lm_layer_name = "lm_LSTM_layer_" + str(i)
        for em_weights, em_gradients in zip(params[em_layer_name], grads[em_layer_name]):
            Weight_SGD(em_weights, em_gradients, batch_size, learn_rate=lstm_learn_rate,
                       clip_norm=lstm_clip_norm)
        for lm_weights, lm_gradients in zip(params[lm_layer_name], grads[lm_layer_name]):
            Weight_SGD(lm_weights, lm_gradients, batch_size, learn_rate=lstm_learn_rate,
                       clip_norm=lstm_clip_norm)


def Construct_batch(lines):
    num_samples = len(lines)
    inputs = []
    targets = []
    in_seq_lens = []
    out_seq_lens = []
    for line in lines:
        ls = line.split("|")
        i = map(int, ls[0].split()[:-1])    # Do not include the <EOS>
        o = map(int, ls[1].split())
        in_seq_lens.append(len(i))
        out_seq_lens.append(len(o))
        inputs.append(i)
        targets.append(o)
    in_batch_len = max(in_seq_lens)
    out_batch_len = max(out_seq_lens)
    in_batch = []
    out_batch = []
    target_batch = []
    for i in range(in_batch_len):
        b = []
        for j in range(num_samples):
            offset = in_batch_len - in_seq_lens[j]
            if i < offset:
                b.append(0)
            else:
                b.append(inputs[j][i - offset])
        in_batch.append(b)
    for i in range(out_batch_len):
        b = []
        b2 = []
        for j in range(num_samples):
            if i < out_seq_lens[j]:
                b.append(targets[j][i])
                if i == 0:
                    b2.append(1)    # 1 for <EOS>
                else:
                    b2.append(targets[j][i - 1])
            else:
                b.append(0)
                b2.append(0)
        target_batch.append(b)
        out_batch.append(b2)
    return in_batch, out_batch, target_batch, in_seq_lens, out_seq_lens


def File_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

# Did not use minibatch
def Train(params, train_file, batch_size=128, epochs = 8, lr_decay=True,
          ff_lr=0.1, lstm_lr=0.7):
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
    line_counts = File_len(train_file)
    total_batch = math.ceil(line_counts / batch_size)
    train_f = open(train_file, 'r')
    while True:
        trained_batch = 0
        log_loss = 0
        trained_words = 0
        t0 = time.time()
        while True:
            lines = []
            for i in range(batch_size):
                line = train_f.readline()
                if not line:
                    train_file.seek(0)
                    trained_epochs += 1
                    trained_batch = 0
                    break
                lines.append(line)
            cur_batch_size = len(lines)
            in_batch, out_batch, target_batch, in_seq_lens, out_seq_lens = Construct_batch(lines)
            # Train the net
            grads, sent_ll = Feed_forward_backward(params, in_batch, out_batch,
                                                   target_batch, in_seq_lens, out_seq_lens)
            All_params_SGD(params, grads, cur_batch_size, ff_learn_rate=ff_lr,
                           lstm_learn_rate=lstm_lr, lstm_clip_norm=5)
            log_loss += sent_ll
            trained_words += (max(in_seq_lens) + max(out_seq_lens)) * cur_batch_size
            # Out put some info
            trained_batch += 1
            if trained_batch % 10 == 0:
                t1 = time.time()
                print "Average sentence log loss: %.5f" % (log_loss / (10.0 * batch_size)),
                print "\tword log loss: %.5f" % (log_loss / float(trained_words)),
                print "\twords/s: %.2f" % (trained_words / float(t1 - t0))
                log_loss = 0
                trained_words = 0
                t0 = t1
            if trained_epochs >= 5:
                # half samples clip
                if trained_batch == total_batch / 2:
                    ff_lr /= 2
                    lstm_lr /= 2
                if trained_batch == total_batch:
                    ff_lr /= 2
                    lstm_lr /= 2
        trained_epochs += 1


def Grad_check(params, data):
    """
    :param params:
    :param sample:
    :return:
    """
    # Create a folder to save the failed gradient check
    if not os.path.exists("debug"):
        os.mkdir("debug")
    else:
        files = os.listdir("debug")
        for f in files:
            os.remove(os.path.join("debug", f))
    print "start gradient check"
    grads, _ = Feed_forward_backward(params, data[0], data[1], data[2],
                                     data[3], data[4])
    Check_diff(params, grads, "W_o", data)
    for i in reversed(range(params["num_layers"])):
        em_layer_name = "em_LSTM_layer_" + str(i)
        lm_layer_name = "lm_LSTM_layer_" + str(i)
        Check_diff(params, grads, em_layer_name, data)
        Check_diff(params, grads, lm_layer_name, data)
    Check_diff(params, grads, "W_we_in", data)
    Check_diff(params, grads, "W_we_out", data)
    print "gradient check end"


def Auto_grad(params, fluct_weights, data):
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
            _, diff1[i, j] = Feed_forward_backward(params, data[0], data[1],
                                                   data[2], data[3], data[4])
            W[i, j] -= 2 * perturbation
            _, diff2[i, j] = Feed_forward_backward(params, data[0], data[1],
                                                   data[2], data[3], data[4])
            # Restore the weight value at (i,j)
            W[i, j] += perturbation
    numerical_grad = (diff1 - diff2) / (2 * perturbation)
    return numerical_grad


def Check_diff(params, grads, name, data):
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
            numDg = Auto_grad(params, weights, data)
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
        numDg = Auto_grad(params, params[name], data)
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
    em_lower_output_acts = Input_feed_forward(params["W_we_in"], in_word_idx_seq)
    init_list = []
    for i in range(params["num_layers"]):
        em_layer_name = "em_LSTM_layer_" + str(i)
        em_weights = params[em_layer_name]
        em_layer_ret = LSTM_feed_forward(em_weights, em_lower_output_acts)
        em_output_acts = em_layer_ret[0]
        em_states = em_layer_ret[1]
        init = [em_output_acts[-1], em_states[-1]]
        init_list.append(init)
        em_lower_output_acts = em_layer_ret[0]
    out_word_idx_seq = [0]  # Generating start from <EOS>
    gen_seq = []
    while True:     # predict 1 time step for each loop
        lm_lower_output_acts = Input_feed_forward(params["W_we_out"], out_word_idx_seq)
        for i in range(params["num_layers"]):
            lm_layer_name = "lm_LSTM_layer_" + str(i)
            lm_weights = params[lm_layer_name]
            lm_layer_ret = LSTM_feed_forward(lm_weights, lm_lower_output_acts,
                                             init_list[i])
            lm_output_acts = lm_layer_ret[0]
            lm_states = lm_layer_ret[1]
            init_list[i] = [lm_output_acts[-1], lm_states[-1]]
            lm_lower_output_acts = lm_layer_ret[0]
        softmax_lower_output_acts = lm_lower_output_acts
        _, predict = Softmax_feed_fordward(params["W_o"],
                                           softmax_lower_output_acts)
        max_idx = np.argmax(predict[0])
        if max_idx == 0:
            break
        out_word_idx_seq = [max_idx]
        gen_seq.append(max_idx)
    return gen_seq


# TODO: change the search to  binary search
def Order_insert(beam_list, node, beam_size):
    if len(beam_list) == 0:
        beam_list.append(node)
        return
    i = 0
    while i < len(beam_list):
        if beam_list[i][2] < node[2]:
            break
        i += 1
    if (i < len(beam_list) or
            (i == len(beam_list) and len(beam_list) < beam_size)):
        beam_list.insert(i, node)
    if len(beam_list) > beam_size:
        beam_list.pop()


# Beam list is decreasing order
# Beam node is a tuple (idx_seq, init, score)
# score is sentence_prob / sentence_words
def Beam_search_generate(params, in_word_idx_seq, beam_size=2, expand_thres=0):
    init_list = []
    beam_list = []
    result_list = []
    em_lower_output_acts = Input_feed_forward(params["W_we_in"], in_word_idx_seq)
    for i in range(params["num_layers"]):
        em_layer_name = "em_LSTM_layer_" + str(i)
        em_weights = params[em_layer_name]
        em_layer_ret = LSTM_feed_forward(em_weights, em_lower_output_acts)
        em_output_acts = em_layer_ret[0]
        em_states = em_layer_ret[1]
        init = [em_output_acts[-1], em_states[-1]]
        init_list.append(init)
        em_lower_output_acts = em_layer_ret[0]
    init_node = ([0], init_list, 0)
    beam_list.append(init_node)
    while beam_size > 0 and len(beam_list) > 0:
        node = beam_list.pop(0)
        if len(node[0]) > 1 and node[0][-1] == 0:
            result_list.append(node[0])
            beam_size -= 1
            continue
        old_gen_seq = node[0]
        old_score = node[2]
        out_word_idx_seq = [old_gen_seq[-1]]
        old_init_list = node[1]

        lm_lower_output_acts = Input_feed_forward(params["W_we_out"], out_word_idx_seq)
        new_init_list = range(params["num_layers"])
        for i in range(params["num_layers"]):
            lm_layer_name = "lm_LSTM_layer_" + str(i)
            lm_weights = params[lm_layer_name]
            lm_layer_ret = LSTM_feed_forward(lm_weights, lm_lower_output_acts,
                                             old_init_list[i])
            lm_output_acts = lm_layer_ret[0]
            lm_states = lm_layer_ret[1]
            new_init_list[i] = [lm_output_acts[-1], lm_states[-1]]
            lm_lower_output_acts = lm_layer_ret[0]
        softmax_lower_output_acts = lm_lower_output_acts
        _, predict = Softmax_feed_fordward(params["W_o"],
                                           softmax_lower_output_acts)
        for j in range(predict[0].shape[0]):
            s = predict[0][j][0]
            if s > expand_thres:
                new_gen_seq = old_gen_seq + [j]
                new_score = (old_score * len(old_gen_seq) + s) / len(new_gen_seq)
                new_node = (new_gen_seq, new_init_list, new_score)
                Order_insert(beam_list, new_node, beam_size)
    return result_list



