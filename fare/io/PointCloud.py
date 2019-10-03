import os
import logging
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from ..geometry.transform import procrustes


class PointCloud(object):
    def __init__(self, file_path=None, points=None, triangles=None, texture=None, tex_coord=None):
        if file_path is None and points is None:
            logging.warning('Cannot file and points are None')
            raise IOError

        self.points = points
        self.coord_ind = triangles
        self.texture = texture
        self.tex_coord = tex_coord

        if file_path:
            ext = os.path.splitext(file_path)[1]
            if ext == '.ply':
                self.load_ply(file_path)
            elif ext == '.wrl':
                self.load_wrl(file_path)
            else:
                raise NotImplementedError

        self.normal = None
        self.normal_f = None

        # check the point cloud

    def compute_normal(self):

        normal = np.zeros(self.points.shape)

        tris = self.points[self.coord_ind]
        # unite normals to the faces
        normal_f = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])

        d = np.sqrt(np.sum(normal_f ** 2, 1)).reshape((-1, 1))
        d[d < 1e-6] = 1

        normal_f /= d
        # unit normal to the vertex
        normal[self.coord_ind[:, 0]] += normal_f
        normal[self.coord_ind[:, 1]] += normal_f
        normal[self.coord_ind[:, 2]] += normal_f

        d = np.sqrt(np.sum(normal ** 2, 1)).reshape((-1, 1))
        d[d < 1e-6] = 1

        normal /= d
        # enforce that the normal are outward
        v = self.points - np.mean(self.points, 1).reshape((-1, 1))
        s = np.sum(v * normal, 0)

        if np.sum(s > 0) < np.sum(s < 0):
            # flip
            normal = -normal
            normal_f = -normal_f

        self.normal = normal
        self.normal_f = normal_f

    def save_ply(self, path):
        try:
            # check the points and triangular
            if self.points is None:
                return False

            if max(self.points.shape) != self.points.shape[0]:
                self.points = self.points.transpose()

            num_vertice = self.points.shape[0]

            f = open(path, 'w')
            f.write('ply\n')
            f.write('format ascii 1.0\n')
            f.write('element vertex ' + str(num_vertice) + '\n')
            f.write('property float x\n')
            f.write('property float y\n')
            f.write('property float z\n')

            if self.texture is not None:
                if max(self.texture.shape) != self.texture.shape[0]:
                    self.texture = self.texture.transpose()

                f.write('property uchar red\n')
                f.write('property uchar green\n')
                f.write('property uchar blue\n')

            if self.coord_ind is not None:
                if max(self.coord_ind.shape) != self.coord_ind.shape[0]:
                    self.coord_ind = self.coord_ind.transpose()
                num_face = self.coord_ind.shape[0]
                f.write('element face ' + str(num_face) + '\n')
                f.write('property list uchar int vertex_indices\n')

            f.write('end_header\n')

            if self.texture is not None:
                for p, t in zip(self.points, self.texture):
                    f.write('%0.4f %0.4f %0.4f %d %d %d\n' % (p[0], p[1], p[2], t[0], t[1], t[2]))
            else:
                for p in self.points:
                    f.write('%0.4f %0.4f %0.4f\n' % (p[0], p[1], p[2]))

            if self.coord_ind is not None:
                for tr in self.coord_ind:
                    f.write('3 %d %d %d\n' % (tr[0], tr[1], tr[2]))
            f.close()

            return True
        except IOError as e:
            logging.warning(e)
            return False

    def load_ply(self, path):

        start_idx = -1
        points = []
        tri = []
        texture = []
        with open(path) as f:
            for i, row in enumerate(f):
                if 'end_header' not in row and start_idx < 0:
                    continue
                elif 'end_header' in row:
                    start_idx = i + 1
                    continue

                if start_idx >= 0:
                    values = list(map(float, row.split()))

                    if len(values) == 3:
                        points.append(values)
                    elif len(values) == 4:
                        tri.append([values[1], values[2], values[3]])
                    elif len(values) == 6:
                        points.append([values[1], values[2], values[3]])
                        texture.append([values[4], values[5], values[6]])

        if len(points) > 0:
            self.points = np.array(points)

        if len(tri) > 0:
            self.coord_ind = np.array(tri)

        if len(texture) > 0:
            self.tex_coord = np.array(texture)

    def save_wrl(self, path):
        print('Save point to wrl file')

    def load_wrl(self, path):

        def _find_field_in_wrl(fid, field_name, entry_len):
            entry_values = []
            start_idx = -1
            for i, row in enumerate(fid):
                if field_name in row:
                    start_idx = i
                    continue

                if start_idx >= 0:
                    row = row.replace(',', '')

                    values = row.split()
                    if len(values) != entry_len:
                        start_idx = -1
                    else:
                        entry_values.append(list(map(float, values)))

            return np.array(entry_values)

        with open(path) as f:
            self.points = _find_field_in_wrl(f, 'point', 3)
            self.coord_ind = _find_field_in_wrl(f, 'coordIndex', 4)
            self.texture = _find_field_in_wrl(f, 'point', 2)
            self.tex_coord = _find_field_in_wrl(f, 'texCoordIndex', 4)


class PointCloudRegister(object):
    def __init__(self, moving, fixed, apply_pa=False, moving_lms=None, fixed_lms=None):
        self.moving_points = moving.points
        self.fixed_points = fixed.points

        if apply_pa:
            if moving_lms is not None and fixed_lms is not None:
                d, Z, tform = procrustes(fixed_lms, moving_lms)
                self.moving_points = np.matmul(tform['scale'] * self.moving_points, tform['rotation']) + tform['translation']

    def register_icp(self, max_iter=20, tolerance=1e-4):
        """
        point to point matching using interative closest points
        :param max_iter: maximum number of iteration
        :param tolerance: tolerance to stop algorithm
        :return:
        """
        point_a = self.moving_points
        point_b = self.fixed_points

        pre_rmse = 0
        pvmse = None

        for it in range(max_iter):
            # find the correspondence
            nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(point_b)
            dist, ind = nbrs.kneighbors(point_a)

            # estimate transformation given correspondence
            point_c = point_b[ind.flatten()]
            # point_c = point_c - np.mean(point_c, axis=0)
            # point_a = point_a - np.mean(point_a, axis=0)

            # f = plt.figure()
            # ax = Axes3D(f)
            # ax.scatter(point_a[:, 0], point_a[:, 1], point_a[:, 2], c='g')
            # ax.scatter(point_c[:, 0], point_c[:, 1], point_c[:, 2], c='r')
            # plt.show()

            # Update the transformation
            M = cv2.estimateAffine3D(point_a, point_c)
            # compute new location
            point_a = np.hstack((point_a, np.ones((point_a.shape[0], 1))))
            point_a = np.dot(point_a, M[1].transpose())

            # f = plt.figure()
            # ax = Axes3D(f)
            # ax.scatter(point_c[:, 0], point_c[:, 1], point_c[:, 2], c='r')
            # ax.scatter(point_a[:, 0], point_a[:, 1], point_a[:, 2], c='g')
            # plt.show()

            # print('debug')
            # compute rmse
            err = np.sum(np.square(point_a - point_c), axis=1)
            rmse = np.sqrt(np.sum(err) / len(err))

            if abs(rmse - pre_rmse) < tolerance:
                pre_rmse = rmse
                pvmse = err / 3
                break
            else:
                pre_rmse = rmse
                pvmse = err / 3

            # print(rmse)

        return pvmse, pre_rmse
