"""
Metric protocols
"""
import numpy as np
from sklearn.metrics import auc
from .metric_funcs import cal_accuracy, cal_roc, cal_pr, cal_iet, cal_cmc, cal_roc_eer, cal_interpolation


class BiometricProtocol(object):
    def __init__(self, template_a_labels, template_b_labels, similarity):
        """
        Superclass for Biometric protocols
        :param template_a_labels: the list of template identities
        :param template_b_labels: the list of template identities
        :param similarity: similarity matrix
        """
        if template_a_labels is None or type(template_a_labels) == np.ndarray or len(template_a_labels) == 0:
            self.template_a_labels = template_a_labels
        else:
            self.template_a_labels = np.array(template_a_labels)

        if template_b_labels is None or type(template_b_labels) == np.ndarray or len(template_b_labels) == 0:
            self.template_b_labels = template_b_labels
        else:
            self.template_b_labels = np.array(template_b_labels)

        self.similarity = similarity if type(similarity) == np.ndarray else np.array(similarity)
        self._binary_labels = None


class BiometricCompareProtocol(BiometricProtocol):
    def __init__(self, template_binary_labels, similarity):
        super(BiometricCompareProtocol, self).__init__([], [], similarity)
        self._binary_labels = template_binary_labels
        self.curves = {'ROC': {}, 'PR': {}, 'ACC': {}, 'OTHERS': {}}

    def compute_roc(self):
        fpr, tpr, thresh = cal_roc(self._binary_labels, self.similarity)
        self.curves['ROC']['FPR'] = fpr
        self.curves['ROC']['TPR'] = tpr
        self.curves['ROC']['Thresh'] = thresh

        min_fpr = min(fpr)
        sel_fpr = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
        sel_fpr = [v for v in sel_fpr if v > min_fpr]
        sel_tpr = cal_interpolation(fpr, tpr, sel_fpr)

        for fpr_v, tpr_v in zip(sel_fpr, sel_tpr):
            self.curves['ROC']['FPR@%s-TPR=' % fpr_v] = tpr_v

    def compute_pr(self):
        precision, recall, thresh = cal_pr(self._binary_labels, self.similarity)
        self.curves['PR']['Precision'] = precision
        self.curves['PR']['Recall'] = recall
        self.curves['PR']['Thresh'] = thresh

    def compute_accuracy(self):
        if len(self.curves['ROC']['Thresh']) == 0:
            self.compute_roc()

        accs = np.zeros((len(self.curves['ROC']['Thresh']),))
        for i, thresh in enumerate(self.curves['ROC']['Thresh']):
            accs[i] = cal_accuracy(self._binary_labels, self.similarity, thresh)
        self.curves['ACC']['Thresh'] = self.curves['ROC']['Thresh']
        self.curves['ACC']['Acc'] = accs

    def compute_eer(self):
        if len(self.curves['ROC']['Thresh']) == 0:
            self.compute_roc()

        eer = cal_roc_eer(self.curves['ROC']['FPR'], self.curves['ROC']['TPR'])
        self.curves['OTHERS']['EER'] = eer

    def compute_auc_roc(self):
        if len(self.curves['ROC']['Thresh']) == 0:
            self.compute_roc()

        auc_roc = auc(self.curves['ROC']['FPR'], self.curves['ROC']['TPR'])

        self.curves['OTHERS']['AUC_ROC'] = auc_roc

    def compute_auc_prc(self):
        if len(self.curves['PR']['Thresh']) == 0:
            self.compute_pr()

        auc_prc = auc(self.curves['PR']['Recall'], self.curves['PR']['Precision'])

        self.curves['OTHERS']['AUC_PR'] = auc_prc


class BiometricCloseSearchProtocol(BiometricProtocol):
    def __init__(self, template_a_labels, template_b_labels, similarity):
        super(BiometricCloseSearchProtocol, self).__init__(template_a_labels, template_b_labels, similarity)

        self.curves = {'CMC': []}

    def compute_cmc_curve(self):
        cmc = cal_cmc(self.template_a_labels, self.template_b_labels, self.similarity)
        self.curves['CMC'] = cmc


class BiometricOpenSearchProtocol(BiometricProtocol):
    def __init__(self, gallery_labels, probe_labels, similarity):
        super(BiometricOpenSearchProtocol, self).__init__(gallery_labels, probe_labels, similarity)

        self.curves = {'DET': {}, 'CMC': [], 'IET': {}}

    def compute_cmc_curve(self):
        self.curves['CMC'] = cal_cmc(self.template_a_labels, self.template_b_labels, self.similarity)

    def compute_iet_curve(self):
        fpir, fnir, thresh = cal_iet(self.template_a_labels, self.template_b_labels, self.similarity)

        self.curves['IET']['FPIR'] = fpir
        self.curves['IET']['FNIR'] = fnir
        self.curves['IET']['Thresh'] = thresh

        min_fpir = min(fpir)
        sel_fpir = [1e-4, 1e-3, 1e-2, 1e-1]
        sel_fpir = [v for v in sel_fpir if v > min_fpir]
        sel_fnir = cal_interpolation(fpir, fnir, sel_fpir)

        for fpir_v, fnir_v in zip(sel_fpir, sel_fnir):
            self.curves['IET']['FPIR@%s-TPIR=' % fpir_v] = 1 - fnir_v
