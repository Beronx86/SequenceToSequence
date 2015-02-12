__author__ = 'v-penlu'


import SequenceToSequence as STS

# text vocab range(10)
in_vocab_size = 10
out_vocab_szie = 14
hidden_size_list = [11]
we_size = 9
params = STS.Construct_net(hidden_size_list, we_size, in_vocab_size, out_vocab_szie)
sample = [[7, 6, 5, 4, 3, 2, 1], [0, 8, 9, 2, 3], [8, 9, 2, 3, 0]]
STS.Grad_check(params, sample)