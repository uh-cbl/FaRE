"""
The basic class of the alignment
"""
import cv2
import numpy as np
import skimage.transform as transformer


class Alignment(object):
    def __init__(self, im_dim):
        self.im_dim = im_dim

    def align(self, im, landmarks):
        raise NotImplementedError


# Template using 5 landmarks:
# (right_eye, left_eye, nose_tip, right_mouth_corner, left_mouth_corner)
TEMPLATE_5LM = np.array([[0.32, 0.45], [0.68, 0.45], [0.5, 0.67], [0.35, 0.85], [0.65, 0.85]])

# Template using 7 landmarks:
# (right_eye_out, right_eye_in, left_eye_in, left_eye_out, nose_tip, right_mouth_corner, left_mouth_corner)
TEMPLATE_7LM = np.array([[0.2, 0.45], [0.4, 0.45], [0.6, 0.45], [0.8, 0.45], [0.5, 0.67], [0.35, 0.85], [0.65, 0.85]])

TEMPLATE_2D = {5: TEMPLATE_5LM,
               7: TEMPLATE_7LM,
               }


class Align2D(Alignment):
    """
    The class to perform 2D Alignment
    """
    def __init__(self, im_dim, n_landmarks=5):
        super(Align2D, self).__init__(im_dim)

        self.n_landmarks = n_landmarks
        self.dst = TEMPLATE_2D[self.n_landmarks] * im_dim
        self.transformer = transformer.SimilarityTransform()

    def align(self, im, landmarks):

        src = landmarks.astype(np.float32)
        self.transformer.estimate(src, self.dst)
        M = self.transformer.params[0:2, :]

        thumbnail = cv2.warpAffine(im, M, (self.im_dim, self.im_dim), borderValue=0)

        return thumbnail


class Align3D(Alignment):
    def __init__(self, im_dim):
        super(Align3D, self).__init__(im_dim)

    def align(self, im, landmarks):
        raise NotImplementedError
