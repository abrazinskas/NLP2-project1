import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_lls(lls, labels, title, save_to, plot_tail=None):
    plt.clf()
    iterations = np.arange(len(lls[0]))
    lls = np.array(lls)
    marker = itertools.cycle(('o', 'p', 'v', 's', '8', '>', '^', '<', 'x'))
    for i, ll in enumerate(lls):
        plt.plot(iterations, ll, '-o', label=labels[i], marker=marker.next())
    plt.xticks(iterations)
    plt.xlabel("Iterations")
    plt.ylabel("Log-likelihood")
    plt.legend()
    plt.title(title)
    if plot_tail is not None:
        space = 0.02 * np.min(lls[:, 1:])
        plt.ylim(np.min(lls[:, -plot_tail:])+space, np.max(lls[:, -plot_tail:])-space)
        plt.xlim(iterations[-plot_tail], iterations[-1] + 0.2)
    plt.savefig(save_to, format='eps', dpi=1000)

def plot_ll(ll, title, save_to):
    plt.clf()
    iterations = np.arange(4, len(ll)+4)
    plt.plot(iterations, ll, '-o')
    plt.xticks(iterations)
    plt.xlabel("Iterations")
    plt.ylabel("Log-likelihood")
    plt.title(title)
    plt.savefig(save_to, format='eps', dpi=1000)
    # plt.show()

def plot_aers(aers, labels, title, save_to, plot_tail=None):
    plt.clf()
    iterations = np.arange(1, len(aers[0])+1)
    marker = itertools.cycle(('o', 'p', 'v', 's', '8', '>', '^', '<', 'x'))
    aers = np.array(aers)
    for i, aer in enumerate(aers):
        plt.plot(iterations, aer, '-o', label=labels[i], marker=marker.next())
    plt.xticks(iterations)
    plt.xlabel("Iterations")
    plt.ylabel("AER")
    plt.legend()
    plt.title(title)

    if plot_tail is not None:
        space = 0.02 * np.min(aers[:, 1:])
        plt.ylim(np.min(aers[:, -plot_tail:])-space, np.max(aers[:, -plot_tail:])+space)
        plt.xlim(iterations[-plot_tail], iterations[-1] + 0.2)
    plt.savefig(save_to, format='eps', dpi=1000)

def plot_aer(aer, title, save_to):
    plt.clf()
    iterations = np.arange(1, len(aer)+1)
    plt.plot(iterations, aer, '-o')
    space = 0.02 * np.min(aer[1:])
    plt.ylim(np.min(aer[1:])-space, np.max(aer[1:])+space)
    plt.xticks(iterations)
    plt.xlabel("Iterations")
    plt.ylabel("AER")
    plt.title(title)
    plt.savefig(save_to, format='eps', dpi=1000)

### Experiment data
# ibm1_long_ll = [-195.1554, -100.1269, -87.3002, -83.4074, -82.0862, -81.4965, -81.1845, -81.0008, -80.8842, -80.8058, \
#                 -80.7505, -80.7101, -80.6796, -80.6561, -80.6375, -80.6225, -80.6103, -80.6002, -80.5918, -80.5847,
#                 -80.5786]
# ibm1_long_aer = [1.0000, 0.3781, 0.3495, 0.3381, 0.3375, 0.3391, 0.3388, 0.3407, 0.3378, 0.3346, 0.3314, 0.3285, \
#                  0.3285, 0.3285, 0.3282, 0.3292, 0.3292, 0.3311, 0.3321, 0.3321, 0.3330]
# ibm1_ll = ibm1_long_ll[:11]
# ibm1_aer = ibm1_long_aer[:11]
#
# ibm2_long_ll = [-237.5674, -109.4525, -84.6752, -75.4823, -72.7437, -71.9122, -71.5774, -71.4072, -71.3103, -71.2503, \
#                 -71.2104, -71.1826, -71.1626, -71.1477, -71.1363, -71.1274, -71.1203, -71.1145, -71.1098, -71.1060, \
#                 -71.1028]
# ibm2_long_aer = [1.0000, 0.3301, 0.2762, 0.2620, 0.2573, 0.2472, 0.2462, 0.2453, 0.2443, 0.2465, 0.2462, 0.2509, \
#                  0.2528, 0.2519, 0.2519, 0.2519, 0.2519, 0.2519, 0.2528, 0.2528, 0.2528]
# ibm2_ll_init_uniform = ibm2_long_ll[:6]
# ibm2_aer_init_uniform = ibm2_long_aer[:6]
# ibm2_ll_init_ibm1 = [-123.1606, -78.5793, -74.5819, -73.4424, -72.8626, -72.4887]
# ibm2_aer_init_ibm1 = [0.3314, 0.2607, 0.2626, 0.2581, 0.2524, 0.2500]
# ibm2_ll_init_random_1 = [-236.5994, -105.6814, -84.0074, -76.2243, -73.4864, -72.4182]
# ibm2_aer_init_random_1 = [0.8111, 0.4640, 0.3118, 0.2829, 0.2583, 0.2521]
# ibm2_ll_init_random_2 = [-239.3501, -111.9621, -87.5498, -77.1103, -73.7048, -72.4886]
# ibm2_aer_init_random_2 = [0.9026, 0.5445, 0.3594, 0.2970, 0.2696, 0.2637]
# ibm2_ll_init_random_3 = [-246.2055, -116.4578, -92.9020, -80.3164, -74.9606, -73.0373]
# ibm2_aer_init_random_3 = [0.9485, 0.6763, 0.4237, 0.3014, 0.2608, 0.2590]
# ibm2_lls = [ibm2_ll_init_uniform, ibm2_ll_init_ibm1, ibm2_ll_init_random_1, ibm2_ll_init_random_2, \
#             ibm2_ll_init_random_3]
# ibm2_aers = [ibm2_aer_init_uniform, ibm2_aer_init_ibm1, ibm2_aer_init_random_1, ibm2_aer_init_random_2, \
#              ibm2_aer_init_random_3]
# ibm2_labels = ["uniform", "IBM 1", "random 1", "random 2", "random 3"]
#
# # Plot IBM 1 results
# plot_ll(ibm1_ll, "IBM 1 log-likelihood", "ibm1_ll.eps")
# plot_aer(ibm1_aer, "IBM 1 validation AER", "ibm1_aer.eps")
# plot_ll(ibm1_long_ll, "IBM 1 log-likelihood for 20 iterations", "ibm1_long_ll.eps")
# plot_aer(ibm1_long_aer, "IBM 1 validation AER for 20 iterations", "ibm1_long_aer.eps")


#
# # Plot IBM 2 results
# plot_ll(ibm2_long_ll, "IBM 2 log-likelihood uniformly initialized for 20 iterations", "ibm2_long_ll.eps")
# plot_aer(ibm2_long_aer, "IBM 2 validation AER uniformly initialized for 20 iterations", "ibm2_long_aer.eps")
# plot_lls(ibm2_lls, ibm2_labels, "IBM 2 log-likelihood for different types of initialization", "ibm2_compare_lls.eps")
# plot_lls(ibm2_lls, ibm2_labels, "IBM 2 log-likelihood for different types of initialization", "ibm2_compare_lls_tail.eps", plot_tail=3)
# plot_aers(ibm2_aers, ibm2_labels, "IBM 2 AERs for different types of initialization", "ibm2_compare_aers.eps")
# plot_aers(ibm2_aers, ibm2_labels, "IBM 2 AERs for different types of initialization", "ibm2_compare_aers_tail.eps", plot_tail=3)


#
# ibm1_var_alpha_sel_aer=[[0.379147, 0.371783, 0.359465, 0.357074, 0.354187, 0.350386, 0.349758, 0.346341, 0.344760, 0.339863],
#                         [0.379147, 0.373574, 0.362333, 0.359962, 0.359345, 0.357143, 0.355965, 0.352427, 0.347656, 0.347656],
#                         [0.379147, 0.375475, 0.363203, 0.360845, 0.359270, 0.358037, 0.356522, 0.355899, 0.351509, 0.349562],
#                         [0.381043, 0.378327, 0.365110, 0.360845, 0.362152, 0.361886, 0.358454, 0.355899, 0.350536, 0.348588]]
#
# ibm1_var_model_selection_aer = [0.379147, 0.371783, 0.359465, 0.357074, 0.354187, 0.350386, 0.349758, 0.346341, 0.344760,
#                                 0.339863, 0.339530, 0.339530, 0.339530, 0.339530, 0.341153, 0.341153, 0.341153, 0.342131,
#                                 0.34213, 0.342131]
# ibm1_var_model_selection_elbo = [-1801037.45, -1605596.42, -1498987.93, -1434939.31, -1392926.87, -1363610.85, -1342090.85,
#                                  -1326046.84, -1313478.85, -1303466.85, -1295474.85, -1288798.85, -1283322.86, -1278518.86,
#                                 -1274506.86, -1271078.86, -1268002.87
#                                  ]

ibm2_var_model_selection_aer = [0.276163, 0.274757, 0.260059, 0.259332, 0.252725, 0.254726, 0.251992, 0.248751]
ibm2_var_model_selection_elbo = [-1299911.05, -1276955.00, -1253628.44, -1238829.22, -1230593.16, -1214482.70, -1216155.43, -1210645.37]

# IBM 1 alpha selection
# plot_aers(ibm1_var_alpha_sel_aer, labels=["1e-2", "1e-3", "1e-4", "1e-5"], title="IBM1 variational alpha selection",
#           save_to="plots/ibm1_var_alpha_sel_aer.eps")
#
# # IBM 1 model selection
# plot_aer(ibm1_var_model_selection_aer, "IBM 1 variational general model selection", save_to="plots/ibm1_var_model_sel_aer.eps")
# plot_ll(ibm1_var_model_selection_elbo, "IBM 1 variational general model selection", save_to="plots/ibm1_var_model_sel_elbo.eps")


# IBM 2 model selection
plot_aer(ibm2_var_model_selection_aer, "IBM 2 variational general model selection",
         save_to="plots/ibm2_var_model_sel_aer.eps")

plot_ll(ibm2_var_model_selection_elbo, "IBM 2 variational general model selection",
         save_to="plots/ibm2_var_model_sel_elbo.eps")

