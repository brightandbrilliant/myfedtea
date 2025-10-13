import torch
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score


def evaluate_AUC(encoder_model, decoder_model, current_data, pos_edge_index, neg_edge_index, device):
    encoder_model.eval()
    decoder_model.eval()
    with torch.no_grad():
        # GCN 仍然使用 data.edge_index (训练集边) 来生成嵌入
        z = encoder_model(current_data.x, current_data.edge_index)

        pos_pred = decoder_model(z[pos_edge_index[0]], z[pos_edge_index[1]])
        neg_pred = decoder_model(z[neg_edge_index[0]], z[neg_edge_index[1]])

        preds = torch.cat([pos_pred, neg_pred], dim=0)
        labels = torch.cat([torch.ones(pos_pred.size(0)), torch.zeros(neg_pred.size(0))], dim=0).to(device)

        # 计算 AUC-ROC
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()

        auc_score = roc_auc_score(labels_np, preds_np)
        return auc_score


def evaluate_accuracy_recall_f1(encoder_model, decoder_model, current_data, pos_edge_index, neg_edge_index,
                                device, threshold=0.8):
    encoder_model.eval()
    decoder_model.eval()
    with torch.no_grad():
        # 使用训练图结构生成所有节点的嵌入
        z = encoder_model(current_data.x, current_data.edge_index)

        # 获取正负边的预测分数
        pos_pred = decoder_model(z[pos_edge_index[0]], z[pos_edge_index[1]])
        neg_pred = decoder_model(z[neg_edge_index[0]], z[neg_edge_index[1]])

        # 拼接预测分数
        preds = torch.cat([pos_pred, neg_pred], dim=0)

        # 将分数转为标签（通过 threshold）
        pred_labels = (preds > threshold).long().cpu().numpy()

        # 构造真实标签
        true_labels = torch.cat([
            torch.ones(pos_pred.size(0)),
            torch.zeros(neg_pred.size(0))
        ], dim=0).long().cpu().numpy()

        # 计算准确率和召回率
        acc = accuracy_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        pre = precision_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)

        return acc, recall, pre, f1

