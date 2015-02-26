import SequenceToSequence as STS
import pickle


pkl_file = "./train.pkl"
f = open(pkl_file, "rb")
samples = pickle.load(f)
f.close()
in_vocab_size = 160001  # 1 for <UNK>
out_vocab_size = 80001
hidden_size_list = [1000, 1000, 1000, 1000]
we_size = 1000
params = STS.Construct_net(hidden_size_list, we_size, in_vocab_size,
                           out_vocab_size, embedding_range=1)
STS.Train(params, samples)
STS.Save_params(params, "pc4.pkl")
