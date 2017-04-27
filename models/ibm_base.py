import numpy as np
import os, pickle
from pickle import UnpicklingError
from scipy.special import digamma, gammaln


# a parent class for IBM models
class IBM_Base():
    def __init__(self):
        self.eps = 1e-6

    # a general function that either invoke ELBO or log-likelihood
    def compute_objective(self, parallel_corpus):
        if self.training_type == "em":
            return self.__compute_log_likelihood(parallel_corpus)
        else:
            return self.__compute_elbo(parallel_corpus)

    def __compute_log_likelihood(self, parallel_corpus):
        log_likelihood = 0.
        nr_of_sent = 0
        for f_sent, e_sent in parallel_corpus:
            nr_of_sent += 1
            for i, f_w in enumerate(f_sent):
                temp_ll = 0.
                for j, e_w in enumerate(e_sent):
                    temp_ll += self.prob_a(j, i, len(e_sent), len(f_sent)) * self.prob_fr_given_eng[f_w, e_w]
                log_likelihood += np.log(temp_ll + self.eps)
        return log_likelihood / nr_of_sent

    # compute the lower-bound of the log-likelihood
    # I use indices in comments as in my notes
    def __compute_elbo(self, parallel_corpus):
        elbo = 0.
        nr_of_sent = 0
        lambdas = self.expected_counts_fr_and_eng + self.alpha
        # First part
        for f_sent, e_sent in parallel_corpus:
            nr_of_sent += 1
            for i, f_w in enumerate(f_sent):
                posteriors = [self.prob_fr_given_eng[f_w, e_w] for e_w in e_sent] # q(a|\phi)
                posteriors /= (np.sum(posteriors) + self.eps)
                for j, e_w in enumerate(e_sent):
                    # 1. and 3. combined
                    elbo += posteriors[j] * np.log(self.prob_a(j, i, len(e_sent), len(f_sent))/(posteriors[j] + self.eps))
                    # 2.
                    expect = np.log(self.prob_fr_given_eng[f_w, e_w] + self.eps)
                    elbo += posteriors[j] * expect

        # Second part (one involving priors over theta)
        expect = np.log(self.prob_fr_given_eng + self.eps)
        inner_loop = expect * (self.alpha - lambdas) + gammaln(lambdas + self.eps) - gammaln(self.alpha + self.eps)
        outer_loop = gammaln(self.french_vocab_size * self.alpha + self.eps) \
                     - gammaln(np.sum(lambdas, axis=0) + self.eps)
        minus_kl = np.sum(inner_loop) + np.sum(outer_loop)
        return elbo/nr_of_sent + minus_kl

    def train(self, parallel_corpus):
        if self.training_type == "em":
            self.train_em(parallel_corpus)
        else:
            self.train_var(parallel_corpus)

    # assuming that the task is to find the alignment and not to do the actual translation
    def infer_alignment(self, french_sentence, english_sentence):
        alignment = []
        for i, f_w in enumerate(french_sentence):
            alignment.append(np.argmax([self.prob_fr_given_eng[f_w, e_w] for e_w in english_sentence]))
        return alignment

    def save_parameters(self, output_dir, name='params.pkl'):
        print 'writing parameters to %s folder' % output_dir
        f = open(os.path.join(output_dir, name), 'wb')
        for param_name in self.params_to_save:
            pickle.dump([param_name, getattr(self, param_name)], f)
        f.close()
        print 'done'

    def load_parameters(self, file_path):
        f = open(file_path, 'rb')
        print '--------------------------'
        print 'loading parameters'
        while True:
            try:
                name, param = pickle.load(f)
                setattr(self, name, param)  # assuming that all parameters are shared variables
            except (EOFError, UnpicklingError):
                break
        f.close()
        print 'done'
        print '--------------------------'
