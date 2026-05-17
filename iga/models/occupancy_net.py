"""
隐式占用网络（Occupancy Network）- 预训练编码器
用于学习 SE(3)-等变的局部几何特征
"""
from iga.models.vn_networks import *           # 导入 Vector Neurons 网络层
from iga.utils.nn_utils import PositionalEncoder  # 导入位置编码模块


class Encoder(nn.Module):
    """
    Vector Neurons 编码器
    作用：将点云编码为具有 SE(3)-等变性的局部特征
    
    输入:
        x: 可选的输入特征 [N, C]
        points: 点云坐标 [N, 3]
        local_features: 局部几何特征 [N, 9]
        centres: 聚类中心坐标 [K, 3]
        其他参数: 批处理相关的索引和指针
    
    输出:
        x: 编码后的局部特征 [K, C_out, 3] （C_out 维向量特征，每个维度都是 3D 向量）
        pos: 聚类中心位置 [K, 3]
        batch: 批次索引 [K]
    """
    
    def __init__(self, local_nn_dims):
        """
        初始化编码器
        
        参数:
            local_nn_dims: VN 编码器的维度列表，如 [3, 512, 512, 512]
                          输入维度 3（3D坐标），输出维度 512（512维向量特征）
        """
        super(Encoder, self).__init__()
        
        # 局部 VN 编码器：处理每个点的局部特征
        # 输入: [N, C_in, 3], 输出: [N, C_out, 3]
        self.vn_encoder = VNEncoder(local_nn_dims)
        
        # 全局 VN 编码器：聚合邻域信息后进行特征变换
        # 输入: [K, C_out, 3], 输出: [K, C_out, 3]
        self.vn_global = VNEncoder([local_nn_dims[-1], local_nn_dims[-1], local_nn_dims[-1]])
        
        # VNPointNetConv：图卷积层，结合局部和全局编码器
        # aggr='mean': 使用均值聚合邻域特征
        self.conv = VNPointNetConv(self.vn_encoder, global_nn=self.vn_global, aggr='mean')

    def initialise(self, context_length):
        """
        初始化上下文缓冲区（用于 few-shot 推理阶段）
        
        参数:
            context_length: 演示样本数量（上下文长度）
        
        作用:
            创建多个缓冲区存储不同演示样本的特征，支持 few-shot 学习
        """
        self.context_length = context_length
        
        # 物体 A 的上下文缓冲区
        self.contex_a_x = [None] * (self.context_length - 1)      # 特征 [K, C, 3]
        self.contex_a_pos = [None] * (self.context_length - 1)    # 位置 [K, 3]
        self.contex_a_batch = [None] * (self.context_length - 1)  # 批次索引 [K]
        
        # 物体 B 的上下文缓冲区
        self.contex_b_x = [None] * (self.context_length - 1)      # 特征 [K, C, 3]
        self.contex_b_pos = [None] * (self.context_length - 1)    # 位置 [K, 3]
        self.contex_b_batch = [None] * (self.context_length - 1)  # 批次索引 [K]

    def encode_sample(self, data):
        """
        编码多个演示样本（用于 few-shot 推理）
        
        参数:
            data: Batch 数据，包含多个演示样本的点云和特征
                  字段命名规则: pos_a_0, features_a_0 (第0个演示样本的物体A)
                              pos_a_1, features_a_1 (第1个演示样本的物体A)
                              ...
        
        输出:
            embeddings: 字典，包含所有演示样本的编码结果
                        'a_p_x': 物体A参考样本特征 [K, C, 3]
                        'a_p_pos': 物体A参考样本位置 [K, 3]
                        'a_c_x': 物体A上下文样本特征 [K*(context_length-1), C, 3]
                        'b_p_x': 物体B参考样本特征 [K, C, 3]
                        ...
        """
        # 编码第一个演示样本（参考样本，anchor）
        # 输入: data.pos_a_0 [N, 3], data.features_a_0 [N, 9], data.centres_a_0 [K, 3]
        # 输出: a_p_x [K, C, 3], a_p_pos [K, 3], a_p_batch [K]
        a_p_x, a_p_pos, a_p_batch = self.encode(None, data.pos_a_0, data.features_a_0, data.centres_a_0,
                                                data.centre_idx_a_0, data.centres_a_0_ptr, data.centre_idx_a_0_batch,
                                                data.centres_a_0_batch, data.point_idx_a_0,
                                                data.pos_a_0_ptr, data.point_idx_a_0_batch)

        # 编码物体 B 的参考样本
        # 输入: data.pos_b_0 [N, 3], data.features_b_0 [N, 9], data.centres_b_0 [K, 3]
        # 输出: b_p_x [K, C, 3], b_p_pos [K, 3], b_p_batch [K]
        b_p_x, b_p_pos, b_p_batch = self.encode(None, data.pos_b_0, data.features_b_0, data.centres_b_0,
                                                data.centre_idx_b_0, data.centres_b_0_ptr, data.centre_idx_b_0_batch,
                                                data.centres_b_0_batch, data.point_idx_b_0,
                                                data.pos_b_0_ptr, data.point_idx_b_0_batch)

        # 编码剩余的演示样本（上下文样本）
        for i in range(1, self.context_length):
            # 动态获取第 i 个演示样本的物体 A 数据
            # 输入: getattr(data, f'pos_a_{i}') [N, 3]
            # 输出: self.contex_a_x[i-1] [K, C, 3]
            self.contex_a_x[i - 1], self.contex_a_pos[i - 1], self.contex_a_batch[i - 1] =\
                self.encode(None,
                            getattr(data, f'pos_a_{i}'),
                            getattr(data, f'features_a_{i}'),
                            getattr(data, f'centres_a_{i}'),
                            getattr(data, f'centre_idx_a_{i}'),
                            getattr(data, f'centres_a_{i}_ptr'),
                            getattr(data, f'centre_idx_a_{i}_batch'),
                            getattr(data, f'centres_a_{i}_batch'),
                            getattr(data, f'point_idx_a_{i}'),
                            getattr(data, f'pos_a_{i}_ptr'),
                            getattr(data, f'point_idx_a_{i}_batch'))

            # 动态获取第 i 个演示样本的物体 B 数据
            # 输入: getattr(data, f'pos_b_{i}') [N, 3]
            # 输出: self.contex_b_x[i-1] [K, C, 3]
            self.contex_b_x[i - 1], self.contex_b_pos[i - 1], self.contex_b_batch[i - 1] =\
                self.encode(None,
                            getattr(data, f'pos_b_{i}'),
                            getattr(data, f'features_b_{i}'),
                            getattr(data, f'centres_b_{i}'),
                            getattr(data, f'centre_idx_b_{i}'),
                            getattr(data, f'centres_b_{i}_ptr'),
                            getattr(data, f'centre_idx_b_{i}_batch'),
                            getattr(data, f'centres_b_{i}_batch'),
                            getattr(data, f'point_idx_b_{i}'),
                            getattr(data, f'pos_b_{i}_ptr'),
                            getattr(data, f'point_idx_b_{i}_batch'))

        # 缓存原始批次索引（用于后续处理）
        a_c_batch = torch.cat(self.contex_a_batch, dim=0)  # [K*(context_length-1)]
        b_c_batch = torch.cat(self.contex_b_batch, dim=0)  # [K*(context_length-1)]

        # 调整批次索引：确保不同演示样本在不同批次中（避免混淆）
        for i in range(2, self.context_length):
            # 每个后续演示样本的批次索引 = 当前索引 + 前一个演示样本的最大索引 + 1
            self.contex_a_batch[i - 1] = self.contex_a_batch[i - 1] + torch.max(self.contex_a_batch[i - 2]) + 1
            self.contex_b_batch[i - 1] = self.contex_b_batch[i - 1] + torch.max(self.contex_b_batch[i - 2]) + 1

        # 拼接所有上下文样本
        a_c_x = torch.cat(self.contex_a_x, dim=0)           # [K*(context_length-1), C, 3]
        a_c_pos = torch.cat(self.contex_a_pos, dim=0)       # [K*(context_length-1), 3]
        a_c_batch_demo = torch.cat(self.contex_a_batch, dim=0)  # [K*(context_length-1)]
        b_c_x = torch.cat(self.contex_b_x, dim=0)           # [K*(context_length-1), C, 3]
        b_c_pos = torch.cat(self.contex_b_pos, dim=0)       # [K*(context_length-1), 3]
        b_c_batch_demo = torch.cat(self.contex_b_batch, dim=0)  # [K*(context_length-1)]

        # 返回编码结果字典
        embeddings = {
            'a_p_x': a_p_x,              # 物体 A 参考样本特征 [K, C, 3]
            'a_p_pos': a_p_pos,          # 物体 A 参考样本位置 [K, 3]
            'a_p_batch': a_p_batch,      # 物体 A 参考样本批次 [K]
            'b_p_x': b_p_x,              # 物体 B 参考样本特征 [K, C, 3]
            'b_p_pos': b_p_pos,          # 物体 B 参考样本位置 [K, 3]
            'b_p_batch': b_p_batch,      # 物体 B 参考样本批次 [K]
            'a_c_x': a_c_x,              # 物体 A 上下文样本特征 [K*(L-1), C, 3]
            'a_c_pos': a_c_pos,          # 物体 A 上下文样本位置 [K*(L-1), 3]
            'a_c_batch': a_c_batch,      # 物体 A 上下文原始批次 [K*(L-1)]
            'a_c_batch_demo': a_c_batch_demo,  # 物体 A 上下文调整后批次 [K*(L-1)]
            'b_c_x': b_c_x,              # 物体 B 上下文样本特征 [K*(L-1), C, 3]
            'b_c_pos': b_c_pos,          # 物体 B 上下文样本位置 [K*(L-1), 3]
            'b_c_batch': b_c_batch,      # 物体 B 上下文原始批次 [K*(L-1)]
            'b_c_batch_demo': b_c_batch_demo,  # 物体 B 上下文调整后批次 [K*(L-1)]
        }
        return embeddings

    def encode(self, x, points, local_features, centres, centre_idx, centres_ptr, centre_idx_batch, centres_batch,
               point_idx, points_ptr, point_idx_batch):
        """
        核心编码方法：将点云编码为局部特征
        
        参数:
            x: 输入特征（可为 None）
            points: 点云坐标 [N, 3]
            local_features: 局部几何特征 [N, 9]（相对位置3 + 叉积3 + 中心点3）
            centres: 聚类中心坐标 [K, 3]
            centre_idx: 每个点所属的聚类中心索引 [N]
            centres_ptr: 聚类中心的指针（批处理偏移量）[B+1]
            centre_idx_batch: centre_idx 的批次索引 [N]
            centres_batch: 聚类中心的批次索引 [K]
            point_idx: 点的全局索引 [N]
            points_ptr: 点云的指针（批处理偏移量）[B+1]
            point_idx_batch: point_idx 的批次索引 [N]
        
        输出:
            x: 编码后的局部特征 [K, C_out, 3]
            pos: 聚类中心位置 [K, 3]
            batch: 批次索引 [K]
        """

        # 1. 调整批次索引（处理批处理时的偏移）
        # 输入: centre_idx [N], centres_ptr [B+1], centre_idx_batch [N]
        # 输出: centre_idx [N]（调整后的索引）
        centre_idx = centre_idx + centres_ptr[centre_idx_batch]
        
        # 输入: point_idx [N], points_ptr [B+1], point_idx_batch [N]
        # 输出: point_idx [N]（调整后的索引）
        point_idx = point_idx + points_ptr[point_idx_batch]
        
        # 2. 构建边索引：point_idx → centre_idx（点到聚类中心的连接）
        # 输入: point_idx [N], centre_idx [N]
        # 输出: edge_index [2, N]
        edge_index = torch.stack([point_idx, centre_idx], dim=0)

        x_dst = None  # 目标节点特征（此处不需要，设为 None）

        # 3. 准备输入特征：将局部特征展平
        # 输入: local_features [N, 9]
        # 输出: feat [N, 9]
        feat = local_features.view(points.shape[0], -1)
        
        # 4. VNPointNetConv 前向传播
        # 输入: (feat [N, 9], x_dst None), (points [N, 3], centres [K, 3]), edge_index [2, N]
        # 输出: x [K, C_out, 3]（K个聚类中心的C_out维向量特征）
        x = self.conv((feat, x_dst), (points, centres), edge_index)

        # 5. 类型转换（AMP 混合精度训练需要）
        # 输入: centres [K, 3] (float32)
        # 输出: centres [K, 3] (与 x 相同 dtype)
        centres = centres.to(x.dtype)
        
        # 6. 返回编码结果
        pos, batch = centres, centres_batch
        return x, pos, batch


class Decoder(nn.Module):
    """
    MLP 解码器
    作用：根据局部特征和查询点位置预测占用概率
    
    输入:
        x: 拼接后的特征 [M, C_in]（C_in = VN特征维度 + 位置编码维度）
    
    输出:
        x: 占用概率 logit [M, 1]（经过 sigmoid 后为 0~1 的概率）
    """
    
    def __init__(self, nn_dims):
        """
        初始化解码器
        
        参数:
            nn_dims: 解码器维度列表，如 [575, 512, 512, 256, 1]
                     输入维度 = VN特征维度(512) + 位置编码维度(63) = 575
                     输出维度 = 1（占用概率）
        """
        super().__init__()
        
        # 创建线性层列表
        # 输入: nn_dims[i] → 输出: nn_dims[i+1]
        self.linear_layers = nn.ModuleList([nn.Linear(nn_dims[i], nn_dims[i + 1]) for i in range(len(nn_dims) - 1)])
        
        # GELU 激活函数（近似 tanh 版本，计算更快）
        self.act = nn.GELU(approximate='tanh')

    def forward(self, x):
        """
        解码器前向传播
        
        参数:
            x: 输入特征 [M, C_in]
        
        输出:
            x: 占用概率 logit [M, 1]
        """
        for i, layer in enumerate(self.linear_layers):
            if i == 0 or i == len(self.linear_layers) - 1:
                # 第一层和最后一层：直接线性变换（无残差连接）
                # 输入: x [M, C_i]
                # 输出: x [M, C_{i+1}]
                x = layer(x)
            else:
                # 中间层：残差连接（缓解梯度消失）
                # 输入: x [M, C_i]
                # 输出: x [M, C_i]（维度不变）
                x = x + layer(x)
            
            if i != len(self.linear_layers) - 1:
                # 非最后一层：应用激活函数
                # 输入: x [M, C_{i+1}]
                # 输出: x [M, C_{i+1}]（经过 GELU）
                x = self.act(x)
        
        # 输出: x [M, 1]（占用概率 logit）
        return x


class AutoEncoder(nn.Module):
    """
    自动编码器（Encoder + Decoder）
    作用：通过隐式占用预测任务预训练编码器
    
    训练阶段输入:
        data: Batch 数据
            - points: 点云坐标 [N, 3]
            - local_features: 局部特征 [N, 9]
            - centres: 聚类中心 [K, 3]
            - queries: 查询点 [M, 3]
            - occupancy: 查询点占用标签 [M]
    
    训练阶段输出:
        occupancy: 预测的占用概率 logit [M]
        target_occupancy: 真实占用标签 [M]
    """
    
    def __init__(self, local_nn_dims, local_num_freq=10):
        """
        初始化自动编码器
        
        参数:
            local_nn_dims: 编码器维度列表，如 [3, 512, 512, 512]
            local_num_freq: 位置编码的频率数量，默认 10
        """
        super().__init__()

        # 位置编码器：将 3D 坐标编码为高频特征
        # 输入: [M, 3]（相对位置）
        # 输出: [M, 3*(1 + 2*10)] = [M, 63]
        self.local_position_encoder = PositionalEncoder(3, local_num_freq, log_space=False)

        # VN 编码器
        self.encoder = Encoder(local_nn_dims)

        # 构建解码器维度：反转编码器维度
        local_decoder_dims = local_nn_dims[::-1]  # [3, 512, 512, 512] → [512, 512, 512, 3]
        local_decoder_dims[-1] = 1  # 输出维度设为 1（占用概率）
        # 输入维度 = VN特征维度 + 位置编码维度
        local_decoder_dims[0] = local_nn_dims[-1] + self.local_position_encoder.d_output  # 512 + 63 = 575

        # 创建解码器
        self.local_decoder = Decoder(local_decoder_dims)

    def forward(self, data):
        """
        自动编码器前向传播
        
        参数:
            data: Batch 数据，包含：
                - x: 可选输入特征
                - points: 点云坐标 [N, 3]
                - local_features: 局部特征 [N, 9]
                - centres: 聚类中心 [K, 3]
                - centre_idx, centres_ptr, ...: 批处理索引
                - queries: 查询点 [Q, 3]
                - queries_idx, queries_centre_idx: 查询点索引
                - occupancy: 查询点占用标签 [Q]
        
        输出:
            occupancy: 预测的占用概率 logit [M]（M 是有效的查询点数量）
            target_occupancy: 真实占用标签 [M]
        """
        # 1. 编码点云获取局部特征
        # 输入: data.points [N, 3], data.local_features [N, 9], data.centres [K, 3]
        # 输出: x [K, C_out, 3], pos [K, 3], batch_pos [K]
        x, pos, batch_pos = self.encoder.encode(data.x, data.points, data.local_features, data.centres, data.centre_idx,
                                                data.centres_ptr, data.centre_idx_batch, data.centres_batch,
                                                data.point_idx, data.points_ptr, data.point_idx_batch)
        
        # 2. 均值池化：将 3D 向量特征压缩为标量特征
        # 输入: x [K, C_out, 3]
        # 输出: x [K, C_out]
        x = mean_pool(x, dim=-1)
        
        # 3. 调整查询点的批次索引（处理批处理偏移）
        # 输入: data.queries_idx [M], data.queries_ptr [B+1], data.queries_idx_batch [M]
        # 输出: data.queries_idx [M]（调整后）
        data.queries_idx = data.queries_idx + data.queries_ptr[data.queries_idx_batch]
        
        # 输入: data.queries_centre_idx [M], data.centres_ptr [B+1], data.queries_centre_idx_batch [M]
        # 输出: data.queries_centre_idx [M]（调整后）
        data.queries_centre_idx = data.queries_centre_idx + data.centres_ptr[data.queries_centre_idx_batch]

        # 4. 计算查询点相对于聚类中心的位置编码
        # 输入: data.queries[data.queries_idx] [M, 3], pos[data.queries_centre_idx] [M, 3]
        # 输出: local_queries [M, 63]
        local_queries = self.local_position_encoder(data.queries[data.queries_idx] - pos[data.queries_centre_idx])
        
        # 5. 拼接局部特征和位置编码
        # 输入: x[data.queries_centre_idx] [M, C_out], local_queries [M, 63]
        # 输出: query_x [M, C_out + 63]
        query_x = torch.cat([x[data.queries_centre_idx], local_queries], dim=1)
        
        # 6. 解码器预测占用概率
        # 输入: query_x [M, C_out + 63]
        # 输出: occupancy [M]（squeeze 后去掉最后一维）
        occupancy = self.local_decoder(query_x).squeeze()
        
        # 7. 获取目标占用标签
        # 输入: data.occupancy[data.queries_idx] [M]
        # 输出: target_occupancy [M]
        target_occupancy = data.occupancy[data.queries_idx].squeeze()
        
        return occupancy, target_occupancy
