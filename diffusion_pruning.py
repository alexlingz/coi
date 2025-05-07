import numpy as np
# import igl
import scipy.sparse
import random
from scipy.spatial import Delaunay
import open3d as o3d
import networkx as nx


class Prune:
    def __init__(self):
        self.delta = 0.05
        self.c0 = None
        self.D1 = None
        self.D2 = None
        self.K_ = None
        self.k_ab = None
        self.confidence_scores = None
        self.Newseeds = []
        self.Delay = []

    def initialize(self, point_source, point_target, corres, c_0=0.3):
        """
        Initializes the pruning procedure based on the geodesic data and correspondence pairs.

        Args:
            geo: geodesic data (containing mesh vertices, etc.)
            corres: list of corresponding vertex pairs (tuples)
            c_0: constant for pruning threshold
        """
        self.c0 = c_0


        # radii = [0.005, 0.01, 0.02, 0.04]
        # # 生成source的mesh
        # pcd_source = o3d.geometry.PointCloud()
        # pcd_source.points = o3d.utility.Vector3dVector(point_source)
        # # # 去除离群点（根据点的邻居来判断）
        # # cl, ind = pcd_source.remove_radius_outlier(nb_points=64, radius=0.05)
        # # # 使用去除离群点后的点云进行 Poisson Surface Reconstruction
        # # pcd_source = pcd_source.select_by_index(ind)
        # pcd_source.estimate_normals()
        # mesh_source = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_source, o3d.utility.DoubleVector(radii))
        # # o3d.visualization.draw_geometries([mesh_source])
        # # o3d.visualization.draw_geometries([pcd_source], point_show_normal=True)
        # # 生成target的mesh
        # pcd_target = o3d.geometry.PointCloud()
        # pcd_target.points = o3d.utility.Vector3dVector(point_target)
        # # # 去除离群点（根据点的邻居来判断）
        # # cl, ind = pcd_target.remove_radius_outlier(nb_points=64, radius=0.05)
        # # # 使用去除离群点后的点云进行 Poisson Surface Reconstruction
        # # pcd_target = pcd_target.select_by_index(ind)
        # pcd_target.estimate_normals()
        # mesh_target = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd_target, o3d.utility.DoubleVector(radii))
        # # o3d.visualization.draw_geometries([mesh_target])
        # # o3d.visualization.draw_geometries([pcd_target], point_show_normal=True)

        # Calculate D (diameter of the mesh) using the furthest sampling scheme and Dijkstra's algorithm
        # 不可靠阿，直接用bbox吧
        # self.D1 = self.calc_diameter(point_source)
        # self.D2 = self.calc_diameter(point_target)
        self.D1 = 0.2# self.calc_euclidean_diameter(point_source)
        self.D2 = 0.2# self.calc_euclidean_diameter(point_target)

        # Initialize K matrix (sparse matrix)
        num_corres = len(corres)
        self.K_ = scipy.sparse.lil_matrix((num_corres, num_corres))
        self.k_ab = np.zeros((num_corres, num_corres))

        # Loop through the corresponding pairs to calculate distances and coefficients
        for i in range(num_corres):
            dist1 = np.zeros(point_source.shape[0])
            dist2 = np.zeros(point_target.shape[0])

            # gamma = np.array([[corres[i][0]]]).astype(np.int32)
            # Compute geodesic distances from the source
            # dist1 = igl.heat_geodesic(point_source,np.asarray(mesh_source.triangles).astype("int64") ,0.1, gamma )
            dist1 = self.compute_geodesic_distance_from_point_cloud(point_source,corres[i][0])
            # gamma = np.array([[corres[i][1]]]).astype(np.int32)
            # Compute geodesic distances from the target
            # dist2 = igl.heat_geodesic(point_target,np.asarray(mesh_target.triangles).astype("int64") ,0.1, gamma, )
            dist2 = self.compute_geodesic_distance_from_point_cloud(point_target,corres[i][1])

            for j in range(i+1, num_corres):
                d1 = dist1[corres[j][0]] / 2.0
                d2 = dist2[corres[j][1]] / 2.0

                self.k_ab[i, j] = d1
                self.k_ab[j, i] = d2

                if d1 < self.delta * self.D1 and d2 < self.delta * self.D2:
                    k_ab = min(d2/d1, d1/d2)
                    if k_ab > self.c0:
                        t = ((k_ab - self.c0) ** 2) / ((1 - self.c0) ** 2)
                        self.K_[i, j] = t
                        self.K_[j, i] = t

        # Convert K_ to sparse matrix format
        self.K_ = self.K_.tocsc()

        # Calculate confidence scores for each column in K_
        total_dist = self.K_.sum()
        self.confidence_scores = np.zeros(self.K_.shape[1])
        for i in range(self.K_.shape[1]):
            self.confidence_scores[i] = self.K_[:, i].sum() / total_dist

    def run_pruning(self, corres):
        """
        Runs the pruning procedure to remove low-confidence correspondences.

        Args:
            corres: list of corresponding vertex pairs (tuples)
        """
        # Initialize data structures
        idx = np.argmax(self.confidence_scores)
        is_in_corres = np.ones(len(corres))
        new_corres = [corres[idx]]
        new_corres_idx = [idx]
        self.Newseeds.append(idx)
        is_in_corres[idx] = 0

        temp = np.ones(len(corres))
        while np.sum(temp) > 0:
            temp = self.confidence_scores * is_in_corres
            idx = np.argmax(temp)
            is_in_corres[idx] = 0

            # Update B1, B2, B
            B1 = np.zeros(len(new_corres), dtype=int)
            B2 = np.zeros(len(new_corres), dtype=int)
            B = np.zeros(len(new_corres), dtype=int)

            # Save calculated geodesic distances in new_corres
            geodist_in_new_corres = []
            for i in range(len(new_corres)):
                if idx > new_corres_idx[i]:
                    d1 = self.k_ab[idx, new_corres_idx[i]]
                    d2 = self.k_ab[new_corres_idx[i], idx]
                else:
                    d2 = self.k_ab[idx, new_corres_idx[i]]
                    d1 = self.k_ab[new_corres_idx[i], idx]

                geodist_in_new_corres.append(d1 / d2)

                if d1 < self.delta * self.D1:
                    B1[i] = 1
                if d2 < self.delta * self.D2:
                    B2[i] = 1
                if d1 < self.delta * self.D1 and d2 < self.delta * self.D2:
                    B[i] = 1

            # Check C1, C2, C3 conditions
            if np.sum(B) > 0 and np.sum(B) / max(np.sum(B1), np.sum(B2)) > 0.5:
                flag = 0
                for i in range(len(new_corres)):
                    if B[i] and min(geodist_in_new_corres[i], 1 / geodist_in_new_corres[i]) > self.c0:
                        flag += 1
                if flag == np.sum(B):
                    new_corres.append(corres[idx])
                    new_corres_idx.append(idx)

                    if len(self.Delay) > 0:
                        for j in self.Delay:
                            is_in_corres[j] = 1
                        self.Delay.clear()

            elif np.sum(B1) == 0 and np.sum(B2) == 0:
                flag = 0
                for i in range(len(self.Newseeds)):
                    d1 = self.k_ab[idx, self.Newseeds[i]]
                    d2 = self.k_ab[self.Newseeds[i], idx]
                    if min(d1 / d2, d2 / d1) >= 0.7:
                        flag += 1

                if flag == len(self.Newseeds):
                    new_corres.append(corres[idx])
                    new_corres_idx.append(idx)
                    self.Newseeds.append(idx)

                    if len(self.Delay) > 0:
                        for j in self.Delay:
                            is_in_corres[j] = 1
                        self.Delay.clear()

            else:
                self.Delay.append(idx)

        # Update correspondences
        # corres.clear()
        # corres.extend(new_corres)
        return np.array(new_corres)

    def calc_diameter(self, point):
        # n*3, np.array
        # 用测地线计算diameter
        n = point.shape[0]
        cur_id = random.randint(0, n - 1)
        v0 = cur_id
        v1 = cur_id
        d0, d1, d2 = -1, -1, 0
        while d2 > d0 and d2 > d1:
            all_dijkstra_dist = np.zeros(n)
            # gamma = np.array([cur_id]).astype("int64") 
            # igl.heat_geodesics_solve(data, gamma, all_dijkstra_dist)  # c++写法
            # all_dijkstra_dist = igl.heat_geodesic(point,tri.astype("int64"),0.1,gamma)
            all_dijkstra_dist = self.compute_geodesic_distance_from_point_cloud(point,cur_id)
            max_idx = np.argmax(all_dijkstra_dist)
            d0 = d1
            d1 = d2
            d2 = all_dijkstra_dist[max_idx]
            cur_id = max_idx
        return d1 / 2.0

    def calc_euclidean_diameter(self, point):
        # n*3, np.array
        # 用测地线计算diameter
        n = point.shape[0]
        cur_id = random.randint(0, n - 1)
        v0 = cur_id
        v1 = cur_id
        d0, d1, d2 = -1, -1, 0
        while d2 > d0 and d2 > d1:
            # all_dijkstra_dist = np.zeros(n)
            all_dist = np.sum((point[cur_id][None,:] - point)**2,axis=1)
            max_idx = np.argmax(all_dist)
            d0 = d1
            d1 = d2
            d2 = all_dist[max_idx]
            cur_id = max_idx
        return d1 / 2.0

    def compute_geodesic_distance_from_point_cloud(self,points, start_idx, k=8):
        """
        使用 k 近邻构建点云图并计算地质距离。

        Parameters:
            points: ndarray
                点云数据，形状为 (N, 3)。
            start_idx: int
                起始点索引。
            k: int
                每个点的邻居数，用于构建图。

        Returns:
            geodesic_distances: ndarray
                从起始点到其他点的地质距离，形状为 (N,)。
        """
        from sklearn.neighbors import NearestNeighbors

        N = points.shape[0]
        G = nx.Graph()

        # 使用 k 近邻查找邻接点
        nbrs = NearestNeighbors(n_neighbors=k).fit(points)
        distances, indices = nbrs.kneighbors(points)

        # 构建图
        for i in range(N):
            for j in range(1, k):  # 跳过自身
                G.add_edge(i, indices[i, j], weight=distances[i, j])

        # 计算单源最短路径长度
        distances = nx.single_source_dijkstra_path_length(G, start_idx) 

        # 转换为数组
        geodesic_distances = np.full(N, np.inf)
        for idx, dist in distances.items():
            geodesic_distances[idx] = dist

        if np.any(np.isinf(geodesic_distances)):
            return self.compute_geodesic_distance_from_point_cloud(points,start_idx,k*2)


        return geodesic_distances+ 1e-5