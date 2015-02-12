__author__ = 'v-penlu'


import SequenceToSequence as STS

# text vocab range(10)
in_vocab_size = 10
out_vocab_szie = 14
hidden_size_list = [11, 12]
we_size = 9
params = STS.Construct_net(hidden_size_list, we_size, in_vocab_size)
STS.Save_params(params)
