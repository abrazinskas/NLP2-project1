import numpy as np

class IBM2():

    def __init__(self, french_vocab_size, english_vocab_size, max_jump=10):
        num_allowed_jumps = 2 * max_jump + 1
        self.expected_lexical_counts = np.zeros((french_vocab_size, english_vocab_size))
        self.expected_jump_counts = np.zeros(num_allowed_jumps)

        # Initialize parameters uniformly. We have no prior knowledge. Note that these parameters
        # are not random and therefore EM will run deterministically.
        self.p_f_given_e = np.full((french_vocab_size, english_vocab_size), 1.0 / french_vocab_size)
        self.jump_p = np.full(num_allowed_jumps, 1.0 / num_allowed_jumps)

        self.max_jump = max_jump
        self.epsilon = 1e-6

    # Performs a single E and M step over the entire given dataset.
    def train(self, parallel_corpus):

        # Make sure the expected counts matrices contain only zeros.
        self.expected_lexical_counts.fill(0.)
        self.expected_jump_counts.fill(0.)

        # Perform the expectation (E) step using the current parameters.
        for (french_sentence, english_sentence) in parallel_corpus:

            for j, f_j in enumerate(french_sentence):

                # Compute the posterior probabilities for each possible alignment for this french word.
                posterior_probs = np.zeros(len(english_sentence))
                deltas = np.zeros(len(english_sentence), dtype=int)
                for i, e_i in enumerate(english_sentence):
                    delta = self.delta(j, i, len(french_sentence), len(english_sentence))
                    alignment_prob = self.jump_prob(delta)
                    posterior_probs[i] = alignment_prob * self.p_f_given_e[f_j, e_i]
                    deltas[i] = delta
                posterior_probs /= (np.sum(posterior_probs) + self.epsilon)

                # Compute the expected counts.
                for i, e_i in enumerate(english_sentence):

                    # Consider the case that f_j was generated from e_i. Add to the expected count of this event
                    # occurring, weighted by the posterior probability.
                    self.expected_lexical_counts[f_j, e_i] += posterior_probs[i]
                    self.expected_jump_counts[self.max_jump * 2 - delta % (self.max_jump * 2 + 1)] += posterior_probs[i]

        # Perform the maximization (M) step to update the parameters.
        self.p_f_given_e = self.expected_lexical_counts / (np.sum(self.expected_lexical_counts, \
                axis=0, keepdims=True) + self.epsilon)
        self.jump_p = self.expected_jump_counts / (np.sum(self.expected_jump_counts) + self.epsilon)

    # Computes the marginal log likelihood.
    def compute_log_likelihood(self, parallel_corpus):
        ll = 0.
        num_data_points = 0
        for (french_sentence, english_sentence) in parallel_corpus:
            num_data_points += 1
            for j, f_j in enumerate(french_sentence):
                inner_sum = 0.
                for i, e_i in enumerate(english_sentence):
                    delta = self.delta(j, i, len(french_sentence), len(english_sentence))
                    p_alignment = self.jump_prob(delta)
                    inner_sum += p_alignment * self.p_f_given_e[f_j, e_i]
                ll += np.log(inner_sum + 1e-10)

        return ll / num_data_points

    # TODO I don't like the index calculation here.
    def jump_prob(self, delta):
        return 0. if np.abs(delta) > self.max_jump else self.jump_p[self.max_jump * 2 - delta % (self.max_jump * 2 + 1)]

    def delta(self, french_pos, eng_pos, french_len, eng_len):
        return int(eng_pos - np.floor(french_pos * (float(eng_len) / french_len)))
