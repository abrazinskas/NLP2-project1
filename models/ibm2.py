import numpy as np
from models.ibm_base import IBM_Base


# IBM translation model 2
class IBM2(IBM_Base):
    def __init__(self, french_vocab_size, english_vocab_size, training_type="em", max_jump=10):
        assert training_type in ["em"]
        self.training_type = training_type
        self.num_allowed_jumps = 2 * max_jump + 1
        self.max_jump = max_jump

        # setup parameters
        self.expected_counts_fr_and_eng = np.zeros(shape=[french_vocab_size, english_vocab_size], dtype="float32")
        self.expected_jump_counts = np.zeros(self.num_allowed_jumps)
        self.prob_fr_given_eng = np.ones(shape=[french_vocab_size, english_vocab_size], dtype="float32")
        self.prob_fr_given_eng /= np.sum(self.prob_fr_given_eng, axis=0, keepdims=True)  # normalization
        self.jump_p = np.full(self.num_allowed_jumps, 1.0 / self.num_allowed_jumps)

        self.params_to_save = []
        IBM_Base.__init__(self)


    # one E and M step over corpus
    def train_EM(self, parallel_corpus):
        self.expected_counts_fr_and_eng.fill(0.)
        self.expected_jump_counts.fill(0.)
        # E-step:
        for f_sent, e_sent in parallel_corpus:
            for i, f_w in enumerate(f_sent):
                posteriors = np.full(len(e_sent), 0.)
                deltas = np.full(len(e_sent), 0.)
                for j, e_w in enumerate(e_sent):
                    prob_a, delta = self.prob_a(j, i, len(e_sent), len(f_sent), return_delta=True)
                    posteriors[j] = self.prob_fr_given_eng[f_w, e_w] * prob_a
                    deltas[j] = delta
                posteriors /= (np.sum(posteriors) + self.eps)
                for j, e_w in enumerate(e_sent):
                    self.expected_counts_fr_and_eng[f_w, e_w] += posteriors[j]
                    e_j_c_indx = self.jump_p_index(deltas[j])
                    self.expected_jump_counts[e_j_c_indx] += posteriors[j]
        # M-step:
        self.prob_fr_given_eng = self.expected_counts_fr_and_eng/(np.sum(self.expected_counts_fr_and_eng, axis=0,
                                                                         keepdims=True) + self.eps)
        self.jump_p = self.expected_jump_counts / (np.sum(self.expected_jump_counts) + self.eps)



    def prob_a(self, j, i, eng_sent_size, fre_sent_size, return_delta=False):
        delta = self.delta(j, i, fre_sent_size, eng_sent_size)
        if return_delta:
            return self.jump_prob(delta), delta
        return self.jump_prob(delta)

    # Returns the jump probability of a given delta.
    def jump_prob(self, delta):
        return 0. if np.abs(delta) > self.max_jump else self.jump_p[self.jump_p_index(delta)]

    # Returns the index in the jump_p array of the given delta. Only works for valid values of delta given the
    # max_jump setting.
    def jump_p_index(self, delta):
        return (self.max_jump * 2 - delta) % self.num_allowed_jumps

    # Calculates the delta for a tuple (j, i, n, m).
    def delta(self, french_pos, eng_pos, french_len, eng_len):
        return int(eng_pos - np.floor(french_pos * (float(eng_len) / french_len)))


