"""
Metric functions
"""
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score, roc_curve
from sklearn.metrics.pairwise import paired_distances
import numpy as np


def compute_pair_dist_mask(x, y, x_m, y_m, metric='cosine'):
    """
    Compute pair distance with mask, main designed for the UR2D features
    :param x: features 1: [n_patch, n_features]
    :param y: features 2: [n_patch, n_features]
    :param x_m: mask 1: [n_patch]
    :param y_m: mask 2: [n_patch]
    :param metric: cosine
    :return: distance
    """
    occ = np.logical_and(x_m, y_m).astype(np.int)

    if x.shape[1] == len(x_m) and y.shape[1] == len(y_m):
        x = np.transpose(x)
        y = np.transpose(y)

    dist = paired_distances(x, y, metric=metric)

    valid_mean_dist = np.sum(dist * occ) / (np.sum(occ) + 1e-8)

    return valid_mean_dist


def compute_pairwise_distances_mask(X, Y, X_M, Y_M, metric='cosine'):
    """
    Compute pairwise distances with mask, main designed for the UR2D features, outputs [n_samples1, n_sample2]
    :param X: features 1: [n_samples1, n_patch, n_features]
    :param Y: features 2: [n_samples2, n_patch, n_features]
    :param X_M: mask 1: [n_samples1, n_patch]
    :param Y_M: mask 2: [n_samples2, n_patch]
    :param metric: cosine
    :return: distances [n_sample1, n_sample2]
    """

    assert len(X) == len(X_M), 'X length should keep same with X_M'
    assert len(Y) == len(Y_M), 'Y length should keep same with Y_M'

    distance = np.zeros((len(X), len(Y)))

    for x_id, (x, x_m) in enumerate(zip(X, X_M)):
        for y_id, (y, y_m) in enumerate(zip(Y, Y_M)):
            distance[x_id, y_id] = compute_pair_dist_mask(x, y, x_m, y_m, metric=metric)

    return distance


def compute_paired_distances_mask(X, Y, X_M, Y_M, metric='cosine'):
    """
    Compute the pair distances with mask, main designed for the UR2D features, outputs [n_samples]
    :param X: features 1: [n_samples, n_patch, n_features]
    :param Y: features 2: [n_samples, n_patch, n_features]
    :param X_M: mask 1: [n_samples, n_patch]
    :param Y_M: mask 2: [n_samples, n_patch]
    :param metric: cosine
    :return: distance: [n_samples]
    """
    assert len(X) == len(X_M), 'X length should keep same with X_M'
    assert len(Y) == len(Y_M), 'Y length should keep same with Y_M'

    distance = np.zeros((len(X),))

    for idx, (x, y, x_m, y_m) in enumerate(zip(X, Y, X_M, Y_M)):
        distance[idx] = compute_pair_dist_mask(x, y, x_m, y_m, metric=metric)

    return distance


def cal_accuracy(y, sim_score, threshold):
    """
    Compute the accuracy
    :param y: ground-truth label
    :param sim_score: similarity score
    :param threshold: threshold
    :return: accuracy
    """
    y_hat = np.greater(sim_score, threshold)
    acc = accuracy_score(y, y_hat)
    return acc


def cal_roc_eer(fpr, tpr):
    """
    compute equal error rate for roc:
    https://www.quora.com/How-can-I-understand-the-EER-Equal-Error-Rate-and-why-we-use-it
    :param fpr: false positive rate
    :param tpr: true positive rate
    :return: EER
    """
    idx = np.argmin(np.abs(fpr + tpr - 1))
    return fpr[idx]


def cal_val_far(y_hat, y_preds):
    """
    Compute the VAR and FAR
    :param y_hat: ground-truth label
    :param y_preds: predict label
    :return:
    """
    true_accept = np.sum(np.logical_and(y_hat, y_preds))
    false_accept = np.sum(np.logical_and(y_hat, np.logical_not(y_preds)))
    n_same = np.sum(y_preds)
    n_diff = np.sum(np.logical_not(y_preds))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def cal_binary_cls_curve(y_hat, y_score, thresholds):
    """
    Compute the binary classification curve
    :param y_hat: ground-truth label, [n_sample]
    :param y_score: predict label, [n_sample]
    :param thresholds: thresholds, [n_threshold]
    :return: FPS: [n_threshold], TPS: [n_threshold]
    """
    assert len(y_hat) == len(y_score)

    y_hat = (y_hat == 1)

    n_thresh = len(thresholds)
    fps = np.zeros((n_thresh, ))
    tps = np.zeros((n_thresh, ))

    for i, thresh in enumerate(thresholds):
        y_preds = np.greater(y_score, thresh)
        y_true_preds = np.logical_and(y_preds, True)
        tps[i] = np.sum(np.logical_and(y_hat, y_true_preds))
        fps[i] = np.sum(np.logical_and(np.logical_not(y_hat), y_true_preds))

    return fps, tps


def cal_roc(y_hat, y_score):
    """
    Compute the ROC
    :param y_hat: ground-truth label
    :param y_score: predict label
    :return: ROC
    """
    # thresholds = np.arange(1, -0.001, -0.001)
    #
    # fps, tps = cal_binary_cls_curve(y_hat, y_score, thresholds)
    #
    # fpr = fps / np.sum(np.logical_not(y_hat))
    # tpr = tps / np.sum(y_hat)

    fpr, tpr, thresholds = roc_curve(y_hat, y_score)

    return fpr, tpr, thresholds


def cal_pr(y_hat, y_score):
    """
    calculate the precision and recall curve
    :param y_hat: ground-truth label, [n_sample]
    :param y_score: predicted similarity score, [n_sample]
    :return: [n_sample]
    """
    thresholds = np.arange(1, -0.001, -0.001)
    fps, tps = cal_binary_cls_curve(y_hat, y_score, thresholds)

    pos_idx = tps > 0
    tps = tps[pos_idx]
    fps = fps[pos_idx]
    thresholds = thresholds[pos_idx]

    precision = tps / (tps + fps)
    recall = tps / np.sum(y_hat)

    return precision, recall, thresholds


def convert_openset_subject_label_2_compare_label(label_a, label_b):
    """
    calculate the openset subject label to the compare label
    :param label_a: gallery labels
    :param label_b: probe labels
    :return: the ground-truth label for probe list to identify whether it is in the gallery set
    """
    gt_label = np.zeros(len(label_b))
    for i, label in enumerate(label_b):
        if label in label_a:
            gt_label[i] = 1

    return gt_label


def cal_iet(gallery_labels, probe_labels, sim_mtx):
    """
    calculate decision trade off curve: details can be found in the paper:
    https://www.nist.gov/sites/default/files/documents/2017/11/22/nistir_8197.pdf
    :param gallery_labels: gallery labels, a 1D array with a shape of (L,)
    :param probe_labels: probe labels, a 1D array with a shape of (M,)
    :param sim_mtx: similarity matrix, a 2D array with a shape of (L, M)
    :return:
        fpir: false positive identification rate
        tpir: true positive identification rate
        thresholds: thresholds
    """
    # check the label length
    assert len(gallery_labels) == sim_mtx.shape[0] and len(probe_labels) == sim_mtx.shape[1], \
        'Shape should be consistent with matrix and ground-truth labels'
    # get the ground-truth compare label for probe
    compare_label_gt = convert_openset_subject_label_2_compare_label(gallery_labels, probe_labels)
    unenroll_index = np.where(compare_label_gt == 0)[0]
    enroll_index = np.where(compare_label_gt == 1)[0]
    # calculate maximum score for wrong pairs
    score_max = []
    for ind in unenroll_index:
        score_max.append(max(sim_mtx[:, ind]))

    thresholds = np.array(score_max)

    fpir, fnir = np.zeros(len(thresholds)), np.zeros(len(thresholds))
    for i, thresh in enumerate(thresholds):
        fpir[i] = np.sum(thresholds >= thresh) / len(thresholds)

        is_correct, is_wrong = 0, 0
        for ind in enroll_index:
            cur_sim = max(sim_mtx[:, ind])
            pred_label = gallery_labels[np.argmax(sim_mtx[:, ind])]
            if pred_label == probe_labels[ind] and cur_sim >= thresh:
                is_correct += 1
            else:
                is_wrong += 1

        fnir[i] = is_wrong / len(enroll_index)

    return fpir, fnir, thresholds


# def cal_tpir_at(fpir):
#     assert len(label_a) == sim_mtx.shape[0] and len(label_b) == sim_mtx.shape[1], \
#         'Shape should be consistent with matrix and ground-truth labels'
#     # get the ground-truth compare label for probe
#     compare_label_gt = convert_openset_subject_label_2_compare_label(label_a, label_b)
#     unenroll_index = np.where(compare_label_gt == 0)[0]
#     enroll_index = np.where(compare_label_gt == 1)[0]
#     # calculate maximum score for wrong pairs
#     score_max = []
#     for ind in unenroll_index:
#         score_max.append(max(sim_mtx[:, ind]))
#     score_max = np.array(score_max)
#     min_value = min(score_max)
#     max_value = max(score_max)
#
#     thresholds = np.arange(min_value, max_value, 0.001)
#     fpir, tpir = np.zeros(len(thresholds)), np.zeros(len(thresholds))
#     for i, thresh in enumerate(thresholds):
#         fpir[i] = np.sum(score_max >= thresh) / len(score_max)
#
#         is_correct = 0
#         for ind in enroll_index:
#             cur_sim = max(sim_mtx[:, ind])
#             pred_label = label_a[np.argmax(sim_mtx[:, ind])]
#             if pred_label == label_b[ind] and cur_sim >= thresh:
#                 is_correct += 1
#
#         tpir[i] = is_correct / len(enroll_index)
#
#     return fpir, tpir, thresholds


def cal_cmc(label_a, label_b, sim_mtx):
    """
    calculate the cumulative match characteristic (cmc)
    :param label_a: [n_sample1]
    :param label_b: [n_sample2]
    :param sim_mtx: [n_sample1, n_sample2]
    :return: CMC [n_sample1]
    """
    assert len(label_a) == sim_mtx.shape[0] and len(label_b) == sim_mtx.shape[1], \
        'Shape should be consistant with matrix and ground-truth labels'

    # convert to close set
    compare_label_gt = convert_openset_subject_label_2_compare_label(label_a, label_b)
    sim_mtx = sim_mtx[:, compare_label_gt == 1]
    label_b = label_b[compare_label_gt == 1]

    # compute CMC
    g_len = len(label_a)
    p_len = len(label_b)

    mat_p = np.tile(label_b.reshape(1, -1), (g_len, 1))
    mat_g = np.tile(label_a.reshape(-1, 1), (1, p_len))

    mat_d = mat_p == mat_g

    mat_rank = np.zeros((g_len, p_len))

    for i in range(p_len):
        mat_rank[:, i] = (-sim_mtx[:, i]).argsort().argsort()

    ranks = np.zeros(g_len)

    for i in range(g_len):
        mat_rank_ = mat_rank <= i
        ranks[i] = np.count_nonzero(mat_d.reshape((-1)) * mat_rank_.reshape((-1))) / p_len

    return ranks


def cal_interpolation(x, y, x_new):
    """
    Interpolate the line with linear interpolation
    :param x: [n_sample]
    :param y: [n_sample]
    :param x_new: [new_n_sample]
    :return: [new_n_sample]
    """
    f = interp1d(x, y)
    if isinstance(x_new, list) or isinstance(x_new, np.ndarray):
        y_new = [f(x) for x in x_new]
    else:
        y_new = [f(x_new)]

    return y_new
