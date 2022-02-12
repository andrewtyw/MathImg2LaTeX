import nltk
import pickle

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(vocab_txt_file_path):
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
    with open(vocab_txt_file_path, 'r') as f:
        for word in f:
            # it is important to read until -1 because we must ignore the '\n'
            vocab.add_word(word[:-1])
    return vocab

if __name__ == "__main__":
    vocab_txt_file_path = "/data/zzengae/tywang/save_model/physics/vocab.txt"
    vocab = build_vocab(vocab_txt_file_path)
    vocab_pkl_file_path = "/data/zzengae/tywang/save_model/physics/vocab.pkl"
    with open(vocab_pkl_file_path, 'wb') as f:
        pickle.dump(vocab, f)
    

