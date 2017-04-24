import numpy as np
from models.ibm_base import IBM_Base
from scipy.special import digamma
np.random.seed(1)

# IBM translation model 1
class IBM1(IBM_Base):
    def __init__(self, french_vocab_size, english_vocab_size, training_type="em"):
        assert training_type in ["em", "var"]
        self.training_type = training_type
        # setup parameters
        self.expected_counts_fr_and_eng = np.zeros(shape=[french_vocab_size, english_vocab_size], dtype="float32")
        # below collection of parameters is used both during E and M steps
        self.prob_fr_given_eng = np.ones(shape=[french_vocab_size, english_vocab_size], dtype="float32")
        # normalization
        self.prob_fr_given_eng /= np.sum(self.prob_fr_given_eng, axis=0, keepdims=True)
        if training_type == "var":
            # Dirichlet's prior parameter (conj to categorical)
            self.alpha = np.float32(np.random.uniform(low=0, high=1.0, size=[french_vocab_size, 1]))
        self.params_to_save = []
        IBM_Base.__init__(self)

    def prob_a(self, j, i, eng_sent_size, fre_sent_size):
        return 1./eng_sent_size

    # one E and M step over corpus
    def train_em(self, parallel_corpus):
            self.expected_counts_fr_and_eng.fill(0.)
            # E-step:
            for f_sent, e_sent in parallel_corpus:
                for i, f_w in enumerate(f_sent):
                    posteriors = [self.prob_fr_given_eng[f_w, e_w] for e_w in e_sent]
                    posteriors /= (np.sum(posteriors) + self.eps)
                    for j, e_w in enumerate(e_sent):
                        self.expected_counts_fr_and_eng[f_w, e_w] += posteriors[j]
            # M-step:
            self.prob_fr_given_eng = self.expected_counts_fr_and_eng/(np.sum(self.expected_counts_fr_and_eng, axis=0,
                                                                             keepdims=True) + self.eps)

    # training via mean-field (variational inference)
    # one E and M step over corpus
    def train_var(self, parallel_corpus):
            self.expected_counts_fr_and_eng.fill(0.)
            # E-step:
            for f_sent, e_sent in parallel_corpus:
                for i, f_w in enumerate(f_sent):
                    posteriors = [self.prob_fr_given_eng[f_w, e_w] for e_w in e_sent]
                    posteriors /= (np.sum(posteriors) + self.eps)
                    for j, e_w in enumerate(e_sent):
                        self.expected_counts_fr_and_eng[f_w, e_w] += posteriors[j]
            # M-step:
            # 1. compute lambdas
            lambdas = self.expected_counts_fr_and_eng + self.alpha
            for f_sent, e_sent in parallel_corpus:
                # 2. compute theta parameters(categorical)
                a = digamma(lambdas[np.ix_(f_sent, e_sent)])
                # TODO: do something smart to avoid summing over all french words
                b = digamma(np.sum(lambdas[:, e_sent], axis=0, keepdims=True))
                self.prob_fr_given_eng[np.ix_(f_sent, e_sent)] = np.exp(a - b)



