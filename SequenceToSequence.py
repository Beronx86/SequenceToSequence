# time orientation prev(ious) futu(re)
# layer orientation lower & upper
#            forward <--> backward
#  input activations <--> output errors
# output activations <--> input errors

import numpy as np

real = np.float32


# Softmax function
def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    return e / np.sum(e)


# Tanh and derivative
def tanh_prime(x):
    y = np.tanh(x)
    y_prime = 1 - (y * y)
    return y, y_prime


# Logistic and derivative
def logistic_prime(x):
    y = 1 / (1 + np.exp(-x))
    y_prime = y * (1 - y)
    return y, y_prime


def LSTM_feed_forward(weights, lower_output_acts, init=[]):
    """Feed forward a LSTM layer. This function also contains the feed forward
       between two layers.
       When this is used to generation, lower_output_acts contains only 1
       time step which is the current step. init contains previous
       configuration. These two things can be used to predict the current
       output.
       :param
       inputs: Output activations of the lower layer.
               It is a matrix. [T, hidden_layer_size]
       weights: A list of matrix [W_iota_y, W_iota_s, W_phi_y, W_phi_s, W,
                                  W_eta_y, W_eta_s, W_o]
               The last column of Weight matrix is bias. So input vectors need
               to append 1.
               input --> gate/cell are weights between two layers.
               W_iota_y     input[t] & hidden[t - 1] --> input gate
               W_iota_s                 state[t - 1] --> input gate (peepholes)
               W_phi_y      input[t] & hidden[t - 1] --> forget gate
               W_phi_s                  state[t - 1] --> forget gate
               W            input[t] & hidden[t - 1] --> cell
               W_eta_y      input[t] & hidden[t - 1] --> output gate
               W_eta_s                  state[t - 1] --> output gate
       init:   The initial previous hidden output activations and
               states. Default value is []. The language model is able to
               include the previous hidden states by this arg.
               It is a list. [hidden output activations, states]
               Both are [layer_size,] vector
       :returns
       output_acts Y: [time_steps, layer_size] matrix
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
    W_iota_y, W_iota_s, W_phi_y, W_phi_s, W, W_eta_y, W_eta_s, W_o = weights
    layer_size = W.shape[0]
    time_steps = lower_output_acts.shape[0]
    input_size = lower_output_acts.shape[1]
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
        prev_hidden = init[0]
        prev_states = init[1]

    for t in range(0, time_steps):
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
        X_iota[t] = W_iota_y.dot(I[t]) + W_iota_s.dot(prev_S)
        Y_iota[t], Yp_iota[t] = logistic_prime(X_iota[t])
        # Calculate forget gate activations
        X_phi[t] = W_phi_y.dot(I[t]) + W_phi_s.dot(prev_S)
        Y_phi[t], Yp_phi[t] = logistic_prime(X_phi[t])
        # Calculate cells
        X[t] = W.dot(I[t])
        G[t], Gp[t] = tanh_prime(X[t])
        S[t] = Y_phi[t] * prev_S + Y_iota * G[t]
        # Calculate output gate activations
        X_eta[t] = W_eta_y.dot(I[t]) + W_eta_s * prev_S
        Y_eta[t], Yp_eta[t] = logistic_prime(X_eta[t])
        # Calculate cell outputs
        H[t], Hp[t] = tanh_prime(S[t])
        Y[t] = Y_eta[t] * H[t]
    return Y, S, H, Hp, G, Gp, Y_eta, Yp_eta, Y_phi, Yp_phi, Y_iota, Yp_iota, I


def LSTM_feed_backward(weights, inter_vars, upper_output_error=[], final=[]):
    W_iota_y, W_iota_s, W_phi_y, W_phi_s, W, W_eta_y, W_eta_s, W_o = weights
    Y, S, H, Hp, G, Gp, Y_eta, Yp_eta, Y_phi, Yp_phi, Y_iota, Yp_iota, I = inter_vars
