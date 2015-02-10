# time orientation prev(ious) futu(re)
# layer orientation lower & upper
#            forward <--> backward
#  input activations <--> output errors
# output activations <--> input errors
# embedding phrase & language model phrase

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
       between two layers. Specifically a lower layer output activations feed
       forward to this layer
       When this is used to generation, lower_output_acts contains only 1
       time step which is the current step. init contains previous
       configuration. These two things can be used to predict the current
       output.

       :param weights:
               A list of matrix [W_iota_y, W_iota_s, W_phi_y, W_phi_s, W,
                                  W_eta_y, W_eta_s, W_o]
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
               It is a matrix. [T, hidden_layer_size]
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


def LSTM_feed_backward(weights, inter_vars, input_errors=0, final=[]):
    """Feed backward a LSTM layer. This function also contains the feed backward
       between two layers. Specifically this layer feed back to the lower layer.
       :param weights:
               A list of matrix [W_iota_y, W_iota_s, W_phi_y, W_phi_s, W,
                                  W_eta_y, W_eta_s, W_o]
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
       :param final:
              [D_hidden_units, D_states, D_forget_gate, D_input_gate]
              The final future hidden unites, states, forget gate, input gate
              derivatives. Default value is []. The embedding phrase is able to
              receive errors from the language model phrase.
              4 terms because:
              S[t - 1] --> input gate[t]
              S[t - 1] --> forget_gate[t]
              S[t - 1] --> S[t]
              H[t - 1] --> Cell[t]
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
    W_iota_y, W_iota_s, W_phi_y, W_phi_s, W, W_eta_y, W_eta_s, W_o = weights
    Y, S, H, Hp, G, Gp, Y_eta, Yp_eta, Y_phi, Yp_phi, Y_iota, Yp_iota, I = inter_vars
    layer_size = W.shape[0]
    time_steps = I.shape[0]
    joint_szie = I.shape[1]
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
        Dl_phi_futu = np.zeros((layer_size, 1), dtype=real)
        Dl_iota_futu = np.zeros((layer_size, 1), dtype=real)
    else:
        # Embedding net receive error from lm net
        Dl_futu, dE_futu, Dl_phi_futu, Dl_iota_futu = final
    # Calculate the error and add it
    for t in reversed(range(time_steps)):
        # Calculate the epsilon
        if input_errors == 0:
            # the top layer of embedding net receives no errors form upper layer
            Eps = W[0: layer_size].T.dot(Dl_futu)
        else:
            Eps = input_errors[t] + W.T[0: layer_size].dot(Dl_futu)
        # Calculate the change in output gates
        Dl_eta = Yp_eta[t] * Eps * H[t]    # element wise multiplication
        Dg_eta_y += Dl_eta.dot(I[t].T)
        Dg_eta_s += Dl_eta * S[t]
        # Calculate the derivative of the error feed back to states
        dE = (Eps * Y_eta[t] * Hp[t] +
              dE_futu * Y_phi[t + 1] +
              Dl_iota_futu * W_iota_s +
              Dl_phi_futu * W_phi_s +
              Dl_eta * W_eta_s)
        # Calculate the delta of the states
        Dl = Y_iota[t] * Gp[t] * dE
        Dg += Dl.dot(I[t].T)
        # Calculate the delta of forget gate
        Dl_phi = Yp_phi[t] * dE * S[t - 1]
        Dg_phi_y += Dl_phi.dot(I[t].T)
        Dg_phi_s += Dl_phi * S[t]
        # Calculate the delta of input gate
        Dl_iota = Yp_iota[t] * dE * G[t]
        Dg_iota_y += Dl_iota.dot(I[t].T)
        Dg_iota_s += Dl_iota * S[t]
        # Save the future ones
        Dl_futu = Dl
        dE_futu = dE
        Dl_phi_futu = Dl_phi
        Dl_iota_futu = Dl_iota
        lower_input_errors[t] = (W_iota_y.T[layer_size: layer_size + input_size].dot(Dl_iota) +
                                 W_phi_y.T[layer_size: layer_size + input_size].dot(Dl_phi) +
                                 W.T[layer_size: layer_size + input_size].dot(Dl) +
                                 W_eta_y.T[layer_size: layer_size + input_size].dot(Dl_eta))
    return (Dg_iota_y, Dg_iota_s, Dg_phi_y, Dg_phi_s, Dg, Dg_eta_y, Dg_phi_s,
            lower_input_errors, [Dl_futu, dE_futu, Dl_phi_futu, Dl_iota_futu])
