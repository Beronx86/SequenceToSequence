import SequenceToSequence as STS
import pickle


pkl_file = "./train.pkl"
f = open(pkl_file, "rb")
print "Loading samples..."
samples = pickle.load(f)
f.close()
in_vocab_size = 160001  # 1 for <UNK>
out_vocab_size = 80001
hidden_size_list = [1000, 1000, 1000]
we_size = 1000
embeding_range = 0.5 / we_size
print "Initializing networks..."
params = STS.Construct_net(hidden_size_list, we_size, in_vocab_size,
                           out_vocab_size, embedding_range=embeding_range)
print "Start training..."
STS.Train(params, samples, lstm_lr=0.1)
print "Saving parameters..."
STS.Save_params(params, "pc4.pkl")
