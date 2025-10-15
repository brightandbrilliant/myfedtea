import torch
import random
from collections import defaultdict


def judge_high_confidence_positive_edges(client_j, u, v, threshold_logit=3.0):
    """
    判断边 (u, v) 是否是客户端 j 视角下的高置信度正样本。

    Args:
        client_j: 知识提供方客户端 j 的对象。
        u, v: 待评估节点的本地索引。
        threshold_logit: 高置信度的 Logit 阈值。
                         Logit >= threshold_logit 意味着 P(边=1) 很高。
                         例如：Logit=2.0 对应 P(边) ≈ 0.88。

    Returns:
        bool: True 如果是高置信度正样本。
    """
    # 确保模型处于评估模式，并关闭梯度追踪
    client_j.encoder.eval()
    client_j.decoder.eval()
    device = client_j.device

    u_tensor = torch.tensor(u, device=device)
    v_tensor = torch.tensor(v, device=device)

    with torch.no_grad():
        # 1. 获取客户端 j 的完整嵌入（注意：每次调用都会重新计算，保持你原有的逻辑）
        z_j = client_j.encoder(client_j.data.x, client_j.data.edge_index)

        # 2. 提取 u 和 v 的嵌入 (假设 u, v 是本地索引)
        z_u = z_j[u_tensor]
        z_v = z_j[v_tensor]

        # 3. 计算 Logit
        logit = client_j.decoder(z_u.unsqueeze(0), z_v.unsqueeze(0)).squeeze()

        # 如果 Logit 大于阈值，则模型 j 强烈认为这是条边
        if logit.item() >= threshold_logit:
            return True, logit.item() # 返回 True 和 Logit 值
        else:
            return False, logit.item()

def extract_augmented_pos_edges(target_fn_types, edge_dict, edge_alignment, client_j, top_k=100):
    candidate_edges = []

    # 1. 根据类型对齐权重抽取候选边
    for (c1_i, c2_i) in target_fn_types:
        aligned_targets = edge_alignment.get((c1_i, c2_i), [])
        for (c1_j, c2_j), weight in aligned_targets:
            # 从 client_j 的正边字典中获取该类型的所有边
            edges_of_type = edge_dict.get((c1_j, c2_j), [])

            # 根据权重计算应抽取的数量
            num_to_select = int(top_k * weight)

            # 简单地将所有边都视为候选，并在后续打乱中随机选择
            # 原始逻辑是取前 K' 个，这里我们简化，全部加入，依赖打乱实现随机性
            # 为了公平性，我们在这里进行一次随机抽样，而不是取前 K' 个

            if edges_of_type:
                # 随机打乱并抽取 num_to_select 条
                random.shuffle(edges_of_type)
                candidate_edges.extend(edges_of_type[:max(1, num_to_select)])  # 至少抽取 1 条以防万一

    # 2. 对所有候选边进行打乱和去重（打乱后，靠前的边更容易被选中）
    unique_candidates = list(set(candidate_edges))
    random.shuffle(unique_candidates)

    # 3. 筛选并选取满足条件的边
    selected_edges = []

    for u, v in unique_candidates:
        if len(selected_edges) >= top_k:
            break

        is_high_conf, _ = judge_high_confidence_positive_edges(
            client_j, u, v
        )

        if is_high_conf:
            selected_edges.append((u, v))

    return selected_edges


def judge_hard_negative_edges(client_j, u, v, threshold=0.5):
    """
        判断边 (u, v) 是否是客户端 j 视角下的困难负样本。

        Args:
            client_j: 知识提供方客户端 j 的对象（包含 encoder, decoder, data）。
            u, v: 待评估节点的全局索引。
            threshold: 困难负样本的概率阈值 P(边=1) > threshold。
                       threshold=0.5 意味着模型 j 预测 logit > 0。

        Returns:
            bool: True 如果是困难负样本。
        """
    # 确保模型处于评估模式，并关闭梯度追踪
    client_j.encoder.eval()
    client_j.decoder.eval()
    device = client_j.device  # 假设 client_j 有一个 device 属性

    u_tensor = torch.tensor(u, device=device)
    v_tensor = torch.tensor(v, device=device)

    with torch.no_grad():
        # 1. 获取客户端 j 的完整嵌入
        # 假设 client_j 的 encoder 接口是 (x, edge_index)
        z_j = client_j.encoder(client_j.data.x, client_j.data.edge_index)

        # 2. 提取 u 和 v 的嵌入
        # 注意：u, v 是在 client_j 数据集中的索引
        z_u = z_j[u_tensor]
        z_v = z_j[v_tensor]

        # 3. 计算 Logit
        # 需要 unsqueeze(0) 以满足 decoder 的 (Batch, Dim) 输入格式
        logit = client_j.decoder(z_u.unsqueeze(0), z_v.unsqueeze(0)).squeeze()

        # 4. 困难度筛选逻辑
        if threshold == 0.5:
            logit_threshold = 0.0
        else:
            logit_threshold = torch.log(torch.tensor(threshold / (1.0 - threshold)))

        # 如果 logit > 阈值，意味着 P(edge) > threshold，模型 j 认为这可能是条边 (难以判断)
        if logit.item() > logit_threshold:
            return True
        else:
            return False


def construct_augmented_neg_edges(aggregated_fp, alignment, cluster_labels_j, pos_edges_j, client_j, top_k=100):
    """
    基于类型对齐和负属性保证，构造增强负边列表，并确保无重复。
    ...
    """
    neg_edge_list = []
    # 用于存储和查重已采样的负边集合
    sampled_neg_edges = set()
    MAX_ATTEMPTS = 500

    for (c1_i, c2_i) in aggregated_fp:
        aligned_targets = alignment.get((c1_i, c2_i), [])

        for (c1_j, c2_j), weight in aligned_targets:
            # !!!此处需要修改: 确保 cluster_labels_j 是 NumPy array 或使用 torch.where，这里是 torch.Tensor
            nodes_c1 = (cluster_labels_j == c1_j).nonzero()[0].tolist()
            nodes_c2 = (cluster_labels_j == c2_j).nonzero()[0].tolist()

            if not nodes_c1 or not nodes_c2:
                continue

            sampled_count = 0
            attempts = 0

            # 注意：这里使用修正后的参数名 top_k
            target_samples = int(top_k * weight) if len(aligned_targets) > 1 else top_k

            while sampled_count < target_samples and attempts < MAX_ATTEMPTS:
                u = random.choice(nodes_c1)
                v = random.choice(nodes_c2)

                edge_tuple = (int(u), int(v))  # 有向边元组

                # 检查条件：
                # 1. u != v
                # 2. edge_tuple 不在客户端 j 的正边集合中（排除正边，只检查当前方向）
                # 3. edge_tuple 不在本次已采样的集合中（排除重复）
                if (u != v and
                        edge_tuple not in pos_edges_j and  # <--- 修正：不再检查 (v, u)
                        edge_tuple not in sampled_neg_edges and
                        judge_hard_negative_edges(client_j, u, v) is True):
                    neg_edge_list.append(edge_tuple)
                    sampled_neg_edges.add(edge_tuple)
                    sampled_count += 1

                attempts += 1

    return neg_edge_list


def build_edge_type_alignment(alignment, nClusters):
    """
    保留边的方向，构建从源边类型 (i,j) 到目标边类型 (i',j') 的映射。
    返回: dict[Tuple[int, int], List[Tuple[Tuple[int, int], float]]]
    """
    edge_mapping = {}

    for i in range(nClusters):
        aligned_i = alignment.get(i, [])  # [(j, score)]
        for j in range(nClusters):
            aligned_j = alignment.get(j, [])  # 保留顺序，不做排序
            key = (i, j)  # 有向边类型

            mapped = []
            for (i_, si) in aligned_i:
                for (j_, sj) in aligned_j:
                    target_key = (i_, j_)
                    weight = si * sj
                    mapped.append((target_key, weight))

            # 合并相同 target_key 的权重
            merged = {}
            for k, w in mapped:
                merged[k] = merged.get(k, 0) + w
            mapped_list = [(k, w) for k, w in merged.items()]
            edge_mapping[key] = mapped_list

    return edge_mapping


def build_positive_edge_dict(data, cluster_labels):
    edge_dict = defaultdict(list)
    edge_index = data.edge_index
    for u, v in edge_index.t().tolist():
        c1, c2 = cluster_labels[u], cluster_labels[v]
        key = (c1, c2)
        edge_dict[key].append((u, v))
    return edge_dict

