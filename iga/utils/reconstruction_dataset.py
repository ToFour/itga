"""
数据预处理模块 - ReconstructDataset
用于为隐式占用网络准备训练数据

核心功能：
1. 加载 ShapeNet 点云数据
2. FPS 采样聚类中心
3. 计算局部几何特征
4. 生成查询点和占用标签
"""
from torch_geometric.data import Dataset, Data  # PyG 数据集和数据类
import torch                                    # PyTorch 核心库
import shutil                                   # 文件操作
import os                                       # 操作系统接口
import pickle                                   # 序列化工具
import numpy as np                              # 数值计算
from tqdm import tqdm                           # 进度条
from torch_geometric.nn import fps, knn, nearest  # PyG 图操作函数
from iga.models.vn_layers import LocalFeatureEncoder  # 局部特征编码器


class ReconstructDataset(Dataset):
    """
    重建数据集类 - 继承自 PyTorch Geometric 的 Dataset
    
    作用：为 Occupancy Network 准备训练数据，包括点云、查询点、占用标签等
    
    输入文件格式：
        原始数据为 pickle 文件，包含：
        - 'points': 点云坐标 [N, 3]
        - 'queries': 查询点坐标 [M, 3]
        - 'dists': 查询点到物体表面的有符号距离 [M]
    
    输出数据结构 (Data 对象)：
        - points: 点云坐标 [N, 3]
        - queries: 查询点坐标 [Q, 3]（包含原始查询点 + 点云点）
        - occupancy: 占用标签 [Q]（0=空，1=占用）
        - centres: FPS 聚类中心 [K, 3]
        - centre_idx: 每个点所属的聚类中心索引 [N]
        - point_idx: 点的全局索引 [N]
        - queries_idx: 查询点的有效索引 [Q]
        - queries_centre_idx: 查询点所属的聚类中心索引 [Q]
        - local_features: 局部几何特征 [N, 3, 3]
    """
    
    def __init__(self, root, transform=None, pre_transform=None, reprocess=True, num_samples=1000,
                 processed_dir='processed', number_local_centers=8, local_neighbours=20, scaling_factor=100):
        """
        初始化数据集
        
        参数：
            root: str - 原始数据目录路径
            transform: callable - 数据转换函数（可选）
            pre_transform: callable - 预处理转换函数（可选）
            reprocess: bool - 是否重新处理数据（True=重新处理，False=使用已处理数据）
            num_samples: int - 使用的样本数量，默认 1000
            processed_dir: str - 处理后数据保存目录，默认 'processed'
            number_local_centers: int - FPS 聚类中心数量，默认 8
            local_neighbours: int - 每个点的局部邻域大小，默认 20
            scaling_factor: int - 坐标缩放因子，默认 100（将米转换为厘米）
        """
        self.scaling_factor = scaling_factor              # 坐标缩放因子
        self.local_feature_encoder = LocalFeatureEncoder(aggr='mean')  # 局部特征编码器
        self.local_neighbours = local_neighbours          # 局部邻域大小
        self.number_local_centers = number_local_centers  # 聚类中心数量
        self.reprocess = reprocess                        # 是否重新处理
        self.processed_ = processed_dir                   # 处理后目录名

        # 获取原始文件列表（筛选包含 'sample' 的文件）
        # 输入: root 目录下的文件列表
        # 输出: self.raw_names - 文件名列表，长度 num_samples
        self.raw_names = [file_name for file_name in os.listdir(root) if 'sample' in file_name][:num_samples]
        self.processed_ = f'{self.processed_}'

        # 如果需要重新处理，删除旧的处理目录
        # 输入: self.reprocess (bool), 处理目录路径
        # 输出: 删除目录（如果存在）
        if self.reprocess and os.path.exists(os.path.join(root, self.processed_)):
            shutil.rmtree(os.path.join(root, self.processed_))

        # 设置数据集长度
        # 输入: self.reprocess (bool)
        # 输出: self.dataset_length - 数据集样本数
        if self.reprocess:
            self.dataset_length = 0  # 重新处理时初始化为 0
        else:
            self.dataset_length = len(os.listdir(os.path.join(root, self.processed_))) - 2  # -2 排除 transform 文件
        
        # 调用父类构造函数
        super(ReconstructDataset, self).__init__(root, transform, pre_transform)
        
        # 更新数据集长度（处理完成后）
        self.dataset_length = len(os.listdir(self.processed_dir)) - 2  # -2 for transforms

    @property
    def raw_dir(self):
        """
        返回原始数据目录
        
        输出: str - 原始数据目录路径（即 root）
        """
        return self.root

    @property
    def raw_file_names(self):
        """
        返回原始文件名列表
        
        输出: list[str] - 原始数据文件名列表
        """
        return self.raw_names

    @property
    def processed_file_names(self):
        """
        返回处理后文件名列表
        
        输出: list[str] - 处理后数据文件名列表
        """
        if self.reprocess:
            return [' ']  # 重新处理时返回空列表占位
        return [f'data_{i}.pt' for i in range(self.len())]

    @property
    def processed_dir(self) -> str:
        """
        返回处理后数据目录
        
        输出: str - 处理后数据目录路径
        """
        return os.path.join(self.root, self.processed_)

    def process(self):
        """
        处理原始数据并保存为 PyTorch Tensor 文件
        
        处理流程：
        1. 加载 pickle 文件
        2. 提取点云和查询点
        3. 生成占用标签
        4. FPS 采样聚类中心
        5. 计算局部特征
        6. 保存处理后数据
        """

        i = 0  # 样本计数器
        # 遍历所有原始文件（带进度条）
        for raw_path in tqdm(self.raw_paths, leave=False):
            # 1. 加载 pickle 文件
            # 输入: raw_path - pickle 文件路径
            # 输出: sample - 字典，包含 'points', 'queries', 'dists'
            sample = pickle.load(open(raw_path, 'rb'))

            # 2. 创建 Data 对象存储数据
            data = Data()
            
            # 3. 提取点云并缩放（将米转换为厘米）
            # 输入: sample['points'] [N, 3] (numpy array)
            # 输出: data.points [N, 3] (torch.Tensor, float32)
            pcd = sample['points'] * self.scaling_factor
            
            # 4. 提取查询点并缩放
            # 输入: sample['queries'] [M, 3] (numpy array)
            # 输出: queries [M, 3] (numpy array)
            queries = sample['queries'] * self.scaling_factor
            
            # 5. 提取有符号距离
            # 输入: sample['dists'] [M] (numpy array)
            # 输出: dist [M] (numpy array)
            dist = sample['dists']

            # 6. 将有符号距离转换为无符号距离（占用预测只关心距离大小）
            # 输入: dist [M] (numpy array, 有符号)
            # 输出: dist [M] (numpy array, 无符号)
            dist = np.abs(dist)
            
            # 7. 构建占用标签：距离 < 0.002（缩放后为 0.2cm）视为占用
            # 输入: dist [M] (numpy array)
            # 输出: occupancy [M, 1] (numpy array, 0 或 1)
            occupancy = np.zeros((queries.shape[0], 1))
            occupancy[dist < 0.002] = 1.  # 距离小于 0.2cm 的点标记为占用
            
            # 8. 将点云点也加入查询点（点云点一定在物体表面，标记为占用）
            # 输入: queries [M, 3], pcd [N, 3]
            # 输出: queries [M+N, 3] (numpy array)
            queries = np.concatenate([queries, pcd], axis=0)
            
            # 输入: occupancy [M, 1], np.ones([N, 1])
            # 输出: occupancy [M+N, 1] (numpy array)
            occupancy = np.concatenate([occupancy, np.ones((pcd.shape[0], 1))], axis=0)
            
            # 9. 转换为 PyTorch Tensor 并存入 Data 对象
            # 输入: pcd [N, 3] (numpy array)
            # 输出: data.points [N, 3] (torch.Tensor, float32)
            data.points = torch.tensor(pcd, dtype=torch.float)
            
            # 输入: queries [M+N, 3] (numpy array)
            # 输出: data.queries [M+N, 3] (torch.Tensor, float32)
            data.queries = torch.tensor(queries, dtype=torch.float)
            
            # 输入: occupancy [M+N, 1] (numpy array)
            # 输出: data.occupancy [M+N] (torch.Tensor, float32, 已 squeeze)
            data.occupancy = torch.tensor(occupancy, dtype=torch.float).squeeze()

            # 10. FPS 采样聚类中心并为每个点分配聚类中心
            # 输入: data.points [N, 3], data.queries [Q, 3], number_local_centers=8
            # 输出: centre_idx [N], point_idx [N], centres [K, 3], queries_idx [Q], queries_centre_idx [Q]
            centre_idx, point_idx, centres, queries_idx, queries_centre_idx = \
                create_clustered_data_sample(data.points,
                                             num_clusters=self.number_local_centers,
                                             queries=data.queries)

            # 11. 将聚类结果存入 Data 对象
            # 输入: centre_idx [N] (torch.Tensor, long)
            # 输出: data.centre_idx [N]
            data.centre_idx = centre_idx
            
            # 输入: point_idx [N] (torch.Tensor, long)
            # 输出: data.point_idx [N]
            data.point_idx = point_idx
            
            # 输入: centres [K, 3] (torch.Tensor, float)
            # 输出: data.centres [K, 3]
            data.centres = centres
            
            # 输入: queries_idx [Q] (torch.Tensor, long)
            # 输出: data.queries_idx [Q]
            data.queries_idx = queries_idx
            
            # 输入: queries_centre_idx [Q] (torch.Tensor, long)
            # 输出: data.queries_centre_idx [Q]
            data.queries_centre_idx = queries_centre_idx

            # 12. 计算局部特征
            # 输入: data.points [N, 3], k=20
            # 输出: id_k_neighbor [2, N*20] (边索引，每行点有 20 个邻居)
            id_k_neighbor = knn(data.points, data.points, k=self.local_neighbours)
            
            # 计算每个点相对于其聚类中心的偏移
            # 输入: data.points [N, 3], data.centres [K, 3], data.centre_idx [N]
            # 输出: data.points - data.centres[data.centre_idx] [N, 3]
            relative_pos = data.points - data.centres[data.centre_idx]
            
            # 提取局部几何特征（相对位置 + 叉积 + 中心点）
            # 输入: relative_pos [N, 3], id_k_neighbor [2, N*20]
            # 输出: features [N, 9] → reshape → [N, 3, 3]
            features = self.local_feature_encoder(relative_pos, id_k_neighbor).view(-1, 3, 3)
            data.local_features = features

            # 13. 保存处理后的数据
            # 输入: data (Data 对象), 保存路径
            # 输出: 保存为 data_{i}.pt 文件
            torch.save(data, os.path.join(self.processed_dir, 'data_{}.pt'.format(i)))
            i += 1

    def len(self):
        """
        返回数据集长度
        
        输出: int - 数据集样本数量
        """
        return self.dataset_length

    def get(self, idx):
        """
        获取指定索引的样本
        
        参数:
            idx: int - 样本索引
        
        输出: Data - 处理后的 Data 对象
        """
        # 输入: idx (int), 文件路径
        # 输出: data (Data 对象)
        data = torch.load(os.path.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


def create_clustered_data_sample(pcd, queries=None, num_clusters=8):
    """
    使用 FPS (Farthest Point Sampling) 采样创建聚类数据样本
    
    核心流程：
    1. FPS 采样选择最远点作为聚类中心
    2. 为每个点分配最近的聚类中心
    3. 构建图边索引（用于图神经网络）
    4. （可选）为查询点分配聚类中心
    
    参数:
        pcd: torch.Tensor - 点云坐标，形状 [N, 3]，N 是点的数量
        queries: torch.Tensor - 查询点坐标，形状 [M, 3]，M 是查询点数量（可选）
        num_clusters: int - 聚类中心数量，默认 8
    
    返回:
        如果 queries 为 None:
            centre_idx: torch.Tensor - 每个点所属的聚类中心索引，形状 [N]，dtype=long
            point_idx: torch.Tensor - 所有点的全局索引，形状 [N]，dtype=long
            centres: torch.Tensor - 聚类中心坐标，形状 [num_clusters, 3]，dtype=float
        如果 queries 不为 None:
            额外返回：
                queries_idx: torch.Tensor - 查询点的全局索引，形状 [M]
                queries_centre_idx: torch.Tensor - 每个查询点所属的聚类中心索引，形状 [M]，dtype=long
    """
    
    ######################################################################################
    # 第 1 步：FPS 采样 - 选择 num_clusters 个最远点作为聚类中心
    # 算法原理：从随机点开始，每次选择离已选点最远的点
    # 输入: pcd [N, 3] (torch.Tensor, float)
    # 输出: idx [num_clusters] (torch.Tensor, long) - 被选中的点在 pcd 中的索引
    
    idx = fps(pcd, ratio=num_clusters / pcd.shape[0])
    # ratio = num_clusters / N，表示采样比例
    # 例如：N=1024, num_clusters=8 → ratio=8/1024=0.0078
    # fps 算法保证采样点尽可能分散覆盖整个点云
    
    # 第 2 步：获取聚类中心的坐标
    # 输入: pcd [N, 3], idx [num_clusters]
    # 输出: cluster_centers [num_clusters, 3] (torch.Tensor, float)
    
    cluster_centers = pcd[idx]
    # 通过索引 idx 从 pcd 中提取聚类中心点坐标
    
    # 第 3 步：为每个点分配最近的聚类中心
    # 使用 k-NN 搜索找到每个点最近的聚类中心
    # 输入: pcd [N, 3], cluster_centers [num_clusters, 3]
    # 输出: cluster_ids [N] (numpy array 或 torch.Tensor, long) - 每个点的最近聚类中心索引
    
    cluster_ids = nearest(pcd, cluster_centers)
    # nearest() 计算每个点到所有聚类中心的距离，返回最近中心的索引
    # 例如：cluster_ids[0]=3 表示第 0 个点属于第 3 个聚类中心
    
    # 第 4 步：构建边索引（用于图神经网络的消息传递）
    # 输入: cluster_ids [N]
    # 输出: centre_idx [N], point_idx [N] (均为 torch.Tensor, long)
    
    centre_idx = torch.tensor(cluster_ids, dtype=torch.long)  # 行索引：每个点属于哪个中心
    point_idx = torch.arange(0, pcd.shape[0], dtype=torch.long)  # 列索引：点的全局索引
    # torch.arange(start=0, end, step=1) → 生成等差数列 [0, 1, 2, ..., N-1]
    # 边索引的含义：
    # edge_index = torch.stack([point_idx, centre_idx]) → [2, N]
    # edge_index[0][i] = point_idx[i] → 源节点（点）
    # edge_index[1][i] = centre_idx[i] → 目标节点（聚类中心）
    # 即：第 point_idx[i] 个点连接到第 centre_idx[i] 个聚类中心
    ######################################################################################
    
    # 第 5 步：将聚类中心转换为 Tensor（确保类型正确）
    # 输入: cluster_centers [num_clusters, 3] (可能是 numpy array 或 Tensor)
    # 输出: centres [num_clusters, 3] (torch.Tensor, float32)
    
    centres = torch.tensor(cluster_centers, dtype=torch.float)
    ######################################################################################
    
    # 第 6 步：如果提供了查询点，为查询点也分配最近的聚类中心
    if queries is not None:
        # 输入: queries [M, 3], cluster_centers [num_clusters, 3]
        # 输出: query_cluster_ids [M] (numpy array 或 torch.Tensor) - 每个查询点的最近聚类中心索引
        
        query_cluster_ids = nearest(queries, cluster_centers)
        # 与点云相同，为每个查询点找到最近的聚类中心
        
        # queries_idx: 查询点的全局索引 [0, 1, 2, ..., M-1]
        # 输入: queries.shape[0] (int)
        # 输出: queries_idx [M] (torch.Tensor, long)
        queries_idx = torch.arange(0, queries.shape[0])
        
        # queries_centre_idx: 每个查询点所属的聚类中心索引
        # 输入: query_cluster_ids [M]
        # 输出: queries_centre_idx [M] (torch.Tensor, long)
        queries_centre_idx = torch.tensor(query_cluster_ids, dtype=torch.long)
        
        # 返回 5 个值
        return centre_idx, point_idx, centres, queries_idx, queries_centre_idx
    
    # 如果没有查询点，返回 3 个值
    return centre_idx, point_idx, centres
    ######################################################################################
