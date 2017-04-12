import numpy as np


# IBM translation model 1
class IBM1():
    def __init__(self, french_vocab_size, english_vocab_size):
        # setup parameters
        self.counts_fr_and_eng = np.zeros([french_vocab_size, english_vocab_size], dtype="float32")
        self.prob_fr_given_eng = np.float32(np.random.uniform(low=0, high=1.0,
                                                              size=[french_vocab_size, english_vocab_size]))
        # normalization
        self.prob_fr_given_eng /= np.sum(self.prob_fr_given_eng, axis=1, keepdims=True)
        self.eps = 1e-6

    # one E and M step over corpora
    def train(self, parallel_corpus):
        # E-step:
        for f_sent, e_sent in parallel_corpus:
            for i, f_w in enumerate(f_sent):
                # pre-compute denom
                denom = np.sum([self.prob_fr_given_eng[f_w, e_w] for e_w in e_sent])
                for j, e_w in enumerate(e_sent):
                    # compute alignment posterior
                    self.counts_fr_and_eng[f_w, e_w] += self.prob_fr_given_eng[f_w, e_w]/(denom + self.eps)
        # M-step:
        self.prob_fr_given_eng = self.counts_fr_and_eng/(np.sum(self.counts_fr_and_eng, axis=1, keepdims=True) + self.eps)

    def compute_log_likelihood(self, parallel_corpus):
        log_likelihood = 0
        for f_sent, e_sent in parallel_corpus:
            for i, f_w in enumerate(f_sent):
                temp_ll = 0
                for j, e_w in enumerate(e_sent):
                    temp_ll += (self.prob_fr_given_eng[f_w, e_w])
                log_likelihood += np.log(temp_ll)
        return log_likelihood
