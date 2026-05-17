from torch_geometric.data import Dataset, Data
import torch
import shutil
import os
import pickle
import numpy as np
from tqdm import tqdm
from torch_geometric.nn import fps, knn, nearest
from iga.models.vn_layers import LocalFeatureEncoder


class ReconstructDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, reprocess=True, num_samples=1000,
                 processed_dir='processed', number_local_centers=8, local_neighbours=20, scaling_factor=100):

        self.scaling_factor = scaling_factor
        self.local_feature_encoder = LocalFeatureEncoder(aggr='mean')
        self.local_neighbours = local_neighbours
        self.number_local_centers = number_local_centers
        self.reprocess = reprocess
        self.processed_ = processed_dir

        self.raw_names = [file_name for file_name in os.listdir(root) if 'sample' in file_name][:num_samples]
        self.processed_ = f'{self.processed_}'

        if self.reprocess and os.path.exists(os.path.join(root, self.processed_)):
            shutil.rmtree(os.path.join(root, self.processed_))

        if self.reprocess:
            self.dataset_length = 0
        else:
            self.dataset_length = len(os.listdir(os.path.join(root, self.processed_))) - 2
        super(ReconstructDataset, self).__init__(root, transform, pre_transform)
        self.dataset_length = len(os.listdir(self.processed_dir)) - 2  # -2 for transforms

    @property
    def raw_dir(self):
        return self.root

    @property
    def raw_file_names(self):
        return self.raw_names

    @property
    def processed_file_names(self):
        if self.reprocess:
            return [' ']
        return [f'data_{i}.pt' for i in range(self.len())]

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.processed_)

    def process(self):

        i = 0
        for raw_path in tqdm(self.raw_paths, leave=False):
            # Read data from `raw_path`.
            sample = pickle.load(open(raw_path, 'rb'))

            data = Data()
            pcd = sample['points'] * self.scaling_factor
            queries = sample['queries'] * self.scaling_factor
            dist = sample['dists']

            # dists are signed, but we want to make them unsigned for the occupancy prediction task.
            dist = np.abs(dist)
            # Construct query points for occupancy net.
            occupancy = np.zeros((queries.shape[0], 1))
            occupancy[dist < 0.002] = 1.
            queries = np.concatenate([queries, pcd], axis=0)
            occupancy = np.concatenate([occupancy, np.ones((pcd.shape[0], 1))], axis=0)
            data.points = torch.tensor(pcd, dtype=torch.float)
            data.queries = torch.tensor(queries, dtype=torch.float)
            data.occupancy = torch.tensor(occupancy, dtype=torch.float).squeeze()

            centre_idx, point_idx, centres, queries_idx, queries_centre_idx = \
                create_clustered_data_sample(data.points,
                                             num_clusters=self.number_local_centers,
                                             queries=data.queries)

            data.centre_idx = centre_idx
            data.point_idx = point_idx
            data.centres = centres
            data.queries_idx = queries_idx
            data.queries_centre_idx = queries_centre_idx

            # Local features
            id_k_neighbor = knn(data.points, data.points, k=self.local_neighbours)
            features = self.local_feature_encoder(data.points - data.centres[data.centre_idx],
                                                  id_k_neighbor).view(-1, 3, 3)
            data.local_features = features

            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def len(self):
        return self.dataset_length

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


def create_clustered_data_sample(pcd, queries=None, num_clusters=8):
    """
    使用 FPS (Farthest Point Sampling) 采样创建聚类数据样本
    
    参数:
        pcd: 点云坐标，形状 [N, 3]，N 是点的数量
        queries: 查询点坐标，形状 [M, 3]，M 是查询点数量（可选）
        num_clusters: 聚类中心数量，默认 8
    
    返回:
        如果 queries 为 None:
            centre_idx: 每个点所属的聚类中心索引，形状 [N]
            point_idx: 所有点的索引，形状 [N]
            centres: 聚类中心坐标，形状 [num_clusters, 3]
        如果 queries 不为 None:
            额外返回 queries_idx, queries_centre_idx
    """
    
    ######################################################################################
    # 第 1 步：FPS 采样 - 选择 num_clusters 个最远点作为聚类中心
    # 输入: pcd [N, 3]
    # 输出: idx [num_clusters] - 被选中的点在 pcd 中的索引
    
    idx = fps(pcd, ratio=num_clusters / pcd.shape[0])
    # ratio = num_clusters / N，表示采样比例
    # 例如：N=1024, num_clusters=8 → ratio=8/1024=0.0078
    # fps 算法保证采样点尽可能分散覆盖整个点云
    
    # 第 2 步：获取聚类中心的坐标
    # 输入: pcd [N, 3], idx [num_clusters]
    # 输出: cluster_centers [num_clusters, 3]
    
    cluster_centers = pcd[idx]
    # 通过索引 idx 从 pcd 中提取聚类中心点坐标
    
    # 第 3 步：为每个点分配最近的聚类中心
    # 输入: pcd [N, 3], cluster_centers [num_clusters, 3]
    # 输出: cluster_ids [N] - 每个点的最近聚类中心索引
    
    cluster_ids = nearest(pcd, cluster_centers)
    # nearest() 计算每个点到所有聚类中心的距离，返回最近中心的索引
    # 例如：cluster_ids[0]=3 表示第 0 个点属于第 3 个聚类中心
    
    # 第 4 步：构建边索引（用于图神经网络的消息传递）
    # 输入: cluster_ids [N]
    # 输出: centre_idx [N], point_idx [N]
    
    centre_idx = torch.tensor(cluster_ids, dtype=torch.long)  # 行索引：每个点属于哪个中心
    point_idx = torch.arange(0, pcd.shape[0], dtype=torch.long)  # 列索引：点的全局索引
    # torch.arange(start=0,end,step=1),生成一个等差数列
    # 边索引的含义：
    # edge_index = [point_idx, centre_idx] 表示 point_idx[i] → centre_idx[i] 的连接
    # 即：第 point_idx[i] 个点属于第 centre_idx[i] 个聚类中心
    ######################################################################################
    
    # 第 5 步：将聚类中心转换为 Tensor
    # 输入: cluster_centers [num_clusters, 3] (可能是 numpy array)
    # 输出: centres [num_clusters, 3] (torch.Tensor)
    
    centres = torch.tensor(cluster_centers, dtype=torch.float)
    ######################################################################################
    
    # 第 6 步：如果提供了查询点，为查询点也分配最近的聚类中心
    if queries is not None:
        # 输入: queries [M, 3], cluster_centers [num_clusters, 3]
        # 输出: query_cluster_ids [M] - 每个查询点的最近聚类中心索引
        
        query_cluster_ids = nearest(queries, cluster_centers)
        # 与点云相同，为每个查询点找到最近的聚类中心
        
        # queries_idx: 查询点的全局索引 [0, 1, 2, ..., M-1]
        queries_idx = torch.arange(0, queries.shape[0])
        
        # queries_centre_idx: 每个查询点所属的聚类中心索引 [M]
        queries_centre_idx = torch.tensor(query_cluster_ids, dtype=torch.long)
        
        # 返回 5 个值
        return centre_idx, point_idx, centres, queries_idx, queries_centre_idx
    
    # 如果没有查询点，返回 3 个值
    return centre_idx, point_idx, centres
    ######################################################################################
