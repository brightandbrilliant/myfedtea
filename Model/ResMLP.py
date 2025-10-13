import torch
import torch.nn as nn
import torch.nn.functional as F


class ResMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        """
        构建一个带有残差连接的MLP解码器，用于链接预测。

        Args:
            input_dim (int): 输入维度，通常是两个GCN嵌入拼接后的维度 (2 * GCN_output_dim)。
            hidden_dim (int): 隐藏层维度。
            num_layers (int): MLP层的总数量 (至少为2，因为需要中间层实现残差)。
            dropout (float): Dropout 比率。
        """
        super(ResMLP, self).__init__()

        if num_layers < 2:
            raise ValueError("num_layers for ResMLPDecoder must be at least 2 for meaningful residual connections.")

        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()

        # 第一层：输入维度 -> 隐藏维度
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # 中间层：隐藏维度 -> 隐藏维度。这些层将应用残差连接。
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # 最后一层：隐藏维度 -> 输出维度（1，表示链接概率的logit）
        self.layers.append(nn.Linear(hidden_dim, 1))

    def forward(self, node_i_embedding: torch.Tensor, node_j_embedding: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            node_i_embedding (torch.Tensor): 节点i的嵌入，形状为 [batch_size, embedding_dim]。
            node_j_embedding (torch.Tensor): 节点j的嵌入，形状为 [batch_size, embedding_dim]。

        Returns:
            torch.Tensor: 链接存在的概率的logits，形状为 [batch_size, 1]。
        """
        # 1. 拼接两个节点的嵌入
        # 例如：如果 embedding_dim 是 64，那么拼接后 input_to_mlp 的形状将是 [batch_size, 128]
        x = torch.cat([node_i_embedding, node_j_embedding], dim=-1)  # 在最后一个维度拼接

        # 2. 经过多层MLP变换
        # 第一层 (Input_dim -> Hidden_dim)
        x = self.layers[0](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 中间层 (Hidden_dim -> Hidden_dim)，应用残差连接
        # 从第二层开始到倒数第二层
        for i in range(1, self.num_layers - 1):
            identity = x  # 保存当前层的输入，作为残差连接的跳跃路径

            linear_layer = self.layers[i]
            x = linear_layer(x)

            # 残差连接：当前层输出 + 上一层输入
            # 由于这里所有中间层的维度都是 hidden_dim，所以维度匹配，可以直接相加
            x = F.relu(x + identity)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 最后一层 (Hidden_dim -> 1)
        # 最后一层不应用激活函数 (ReLU) 和 Dropout，因为我们希望直接输出 logits
        output_logits = self.layers[-1](x)

        return output_logits
