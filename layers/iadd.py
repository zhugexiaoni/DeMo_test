"""
Instance-Aware Dynamic Distillation (IADD) Plugin

实现非平衡多模态学习中的样本级动态蒸馏和难样本挖掘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class IADDPlugin(nn.Module):
    """
    Instance-Aware Dynamic Distillation (IADD) Plugin
    论文核心创新点实现：
    1. MCD (Modality Confidence Differential) 指标计算
    2. 动态双向蒸馏 Loss
    3. 混合距离矩阵 (用于 Triplet Loss)
    """

    def __init__(self, temperature=2.0, hard_neg_k=10, lambda_distill=1.0, lambda_hybrid=0.5):
        super(IADDPlugin, self).__init__()
        self.T = temperature  # Sigmoid 温度系数，控制权重的陡峭程度
        self.k = hard_neg_k   # MCD计算中难负样本的数量
        self.lambda_distill = lambda_distill # 蒸馏 Loss 的权重
        self.lambda_hybrid = lambda_hybrid   # 混合距离的权重
        
        # 基础组件
        self.kl_loss = nn.KLDivLoss(reduction='none')
        self.sigmoid = nn.Sigmoid()

    def compute_mcd(self, features, labels):
        """
        计算模态置信度差分 (MCD)
        features: (B, D) 归一化后的特征
        labels: (B)
        """
        B = features.size(0)
        # 1. 计算相似度矩阵 (B, B)
        sim_matrix = torch.mm(features, features.t())

        # 2. 构建正负样本掩码
        # is_pos: (B, B), 同一ID为True
        is_pos = labels.expand(B, B).eq(labels.expand(B, B).t())
        # 排除自己 (对角线)
        eye = torch.eye(B, device=features.device).bool()
        is_pos = is_pos & (~eye)
        is_neg = ~is_pos

        mcd_scores = torch.zeros(B, device=features.device)

        for i in range(B):
            # --- Part A: 类内紧凑度 (Compactness) ---
            pos_sims = sim_matrix[i][is_pos[i]]
            if len(pos_sims) > 0:
                pos_score = pos_sims.mean()
            else:
                pos_score = torch.tensor(0.0, device=features.device)

            # --- Part B: 类间分离度 (Separability) ---
            neg_sims = sim_matrix[i][is_neg[i]]
            if len(neg_sims) > 0:
                # 只取最难的 Top-K 负样本
                curr_k = min(self.k, len(neg_sims))
                hard_neg_sims, _ = torch.topk(neg_sims, k=curr_k, largest=True)
                neg_score = hard_neg_sims.mean()
            else:
                neg_score = torch.tensor(0.0, device=features.device)

            # --- Part C: 差分 ---
            mcd_scores[i] = pos_score - neg_score

        return mcd_scores

    def get_hybrid_dist_matrix(self, dist_m1, dist_m2, weight_m1):
        """
        计算混合距离矩阵，用于 Hard Mining
        dist_m1, dist_m2: (B, B) 欧氏距离矩阵
        weight_m1: (B) 模态1的权重
        """
        # 将权重扩展为 (B, B)，这里我们假设以 Anchor (行) 的权重为主
        # 也就是说，如果 Anchor 觉得 RGB 准，那找难样本就主要信 RGB 的距离
        W = weight_m1.unsqueeze(1).expand_as(dist_m1)
        
        # 混合距离
        dist_hybrid = W * dist_m1 + (1 - W) * dist_m2
        return dist_hybrid

    def forward(self, logits_m1, logits_m2, feats_m1, feats_m2, labels):
        """
        前向传播
        m1: 通常指 RGB
        m2: 通常指 IR/NI/TI
        """
        B = logits_m1.size(0)
        
        # 0. 特征归一化 (用于MCD计算)
        feats_m1_norm = F.normalize(feats_m1, p=2, dim=1)
        feats_m2_norm = F.normalize(feats_m2, p=2, dim=1)

        # 1. 计算 MCD 指标
        mcd_m1 = self.compute_mcd(feats_m1_norm, labels)
        mcd_m2 = self.compute_mcd(feats_m2_norm, labels)

        # 2. 计算动态权重 alpha (表示 m1 相比 m2 的相对优势)
        # gap > 0 表示 m1 强，alpha -> 1
        # gap < 0 表示 m2 强，alpha -> 0
        gap = mcd_m1 - mcd_m2
        alpha = self.sigmoid(gap * self.T) # shape (B,)

        # 3. 计算双向动态蒸馏 Loss
        # PyTorch KLDivLoss: input=log_prob(Student), target=prob(Teacher)
        log_prob_m1 = F.log_softmax(logits_m1, dim=1)
        log_prob_m2 = F.log_softmax(logits_m2, dim=1)
        prob_m1 = F.softmax(logits_m1, dim=1)
        prob_m2 = F.softmax(logits_m2, dim=1)

        # Case A: m1 教 m2 (权重 alpha)
        # 注意：老师需要 detach，不反向传播梯度到老师
        loss_m1_teach_m2 = self.kl_loss(log_prob_m2, prob_m1.detach()).sum(dim=1) 
        
        # Case B: m2 教 m1 (权重 1-alpha)
        loss_m2_teach_m1 = self.kl_loss(log_prob_m1, prob_m2.detach()).sum(dim=1)

        # 加权求和并取平均
        loss_distill = (alpha * loss_m1_teach_m2 + (1 - alpha) * loss_m2_teach_m1).mean()

        # 4. 准备混合距离矩阵 (给外部 Triplet Loss 使用)
        # 计算欧氏距离矩阵
        dist_m1 = self._euclidean_dist(feats_m1_norm, feats_m1_norm)
        dist_m2 = self._euclidean_dist(feats_m2_norm, feats_m2_norm)
        
        hybrid_dist = self.get_hybrid_dist_matrix(dist_m1, dist_m2, alpha)

        # 返回内容
        return {
            "loss_distill": loss_distill * self.lambda_distill,
            "hybrid_dist": hybrid_dist,
            "mcd_m1": mcd_m1.mean().item(),
            "mcd_m2": mcd_m2.mean().item(),
            "alpha_mean": alpha.mean().item() # 用于监控 RGB 是否长期主导
        }

    def _euclidean_dist(self, x, y):
        """计算欧氏距离矩阵 (x-y)^2 = x^2 + y^2 - 2xy"""
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist

class HybridTripletLoss(nn.Module):
    """
    接受预计算距离矩阵的 Triplet Loss
    """
    def __init__(self, margin=0.3):
        super(HybridTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, dist_mat, labels):
        """
        dist_mat: (B, B) 预计算好的距离矩阵
        """
        n = dist_mat.size(0)
        
        # 构建正负样本掩码
        is_pos = labels.expand(n, n).eq(labels.expand(n, n).t())
        is_neg = ~is_pos
        
        # Hard Mining: 
        # 对于每个 Anchor，找距离最远的正样本 (Hard Positive)
        # 和距离最近的负样本 (Hard Negative)
        
        # Hard Positive: masked fill 0 for negatives, then max
        dist_ap, _ = torch.max(dist_mat * is_pos.float(), dim=1)
        
        # Hard Negative: masked fill inf for positives, then min
        dist_an, _ = torch.min(dist_mat + (is_pos.float() * 1e6), dim=1)
        
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
