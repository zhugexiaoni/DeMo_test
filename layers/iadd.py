"""
Instance-Aware Dynamic Distillation (IADD) Plugin

实现非平衡多模态学习中的样本级动态蒸馏和难样本挖掘
(Robust & Vectorized Version)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class IADDPlugin(nn.Module):
    """
    Instance-Aware Dynamic Distillation (IADD) Plugin
    """

    def __init__(self, temperature=2.0, hard_neg_k=10, lambda_distill=1.0, lambda_hybrid=0.5):
        super(IADDPlugin, self).__init__()
        self.T = temperature
        self.k = hard_neg_k
        self.lambda_distill = lambda_distill
        self.lambda_hybrid = lambda_hybrid
        
        self.kl_loss = nn.KLDivLoss(reduction='none')
        self.sigmoid = nn.Sigmoid()

    def compute_mcd_vectorized(self, features, labels):
        """
        全向量化计算 MCD (修复版)
        """
        B = features.size(0)
        
        # --- DEBUG SECTION START ---
        # if torch.rand(1).item() < 0.1: # Sample 10% of batches to print
        #     with torch.no_grad():
        #         print(f"\n[IADD DEBUG] Batch Size: {B}")
        #         print(f"[IADD DEBUG] Features: Mean={features.mean().item():.4f}, Std={features.std().item():.4f}, Norm={features.norm(dim=1).mean().item():.4f}")
        #         print(f"[IADD DEBUG] Labels (first 10): {labels[:10].tolist()}")
        # --- DEBUG SECTION END ---

        # 1. 相似度矩阵 (B, B) -> 范围 [-1, 1]
        sim_matrix = torch.mm(features, features.t())

        # 2. 构建掩码
        labels = labels.view(B, 1)
        is_pos = torch.eq(labels, labels.t())
        
        # 排除对角线
        eye = torch.eye(B, device=features.device, dtype=torch.bool)
        is_pos = is_pos & ~eye
        is_neg = ~is_pos & ~eye

        # --- DEBUG SECTION START ---
        # if torch.rand(1).item() < 0.1:
        #     pos_pairs = is_pos.sum().item()
        #     neg_pairs = is_neg.sum().item()
        #     print(f"[IADD DEBUG] Positive Pairs: {pos_pairs}, Negative Pairs: {neg_pairs}")
        #     if pos_pairs == 0:
        #         print("[IADD WARNING] No positive pairs found! Check if PK Sampler is working.")
        # --- DEBUG SECTION END ---

        # --- Part A: 类内紧凑度 ---
        # 必须处理 pos_counts 为 0 的情况 (尽管很少见)
        pos_counts = is_pos.sum(dim=1).float()
        pos_sum = (sim_matrix * is_pos.float()).sum(dim=1)
        
        # 避免除以 0
        pos_scores = torch.zeros_like(pos_sum)
        mask_valid_pos = pos_counts > 0
        pos_scores[mask_valid_pos] = pos_sum[mask_valid_pos] / pos_counts[mask_valid_pos]

        # --- Part B: 类间分离度 ---
        # 使用 -2.0 作为填充值 (因为 Cosine Sim 最小是 -1.0)
        neg_matrix_for_topk = sim_matrix.clone()
        neg_matrix_for_topk.masked_fill_(~is_neg, -2.0)

        # 动态调整 K (防止 Batch Size 小于 K)
        actual_k = min(self.k, B - 1)
        if actual_k > 0:
            # Check if we have enough negatives
            neg_counts = is_neg.sum(dim=1)
            # If any sample has fewer negatives than K (very rare in ReID), cap K for that sample?
            # Torch topk requires constant K. So we use global min or fallback.
            # Usually ReID batch size (64) >> K (10) and P*K=4*16. Negatives are plenty.
            
            # Robustness: ensure we don't crash if negs < k
            if neg_counts.min() < actual_k:
                 actual_k = neg_counts.min().item()

            if actual_k > 0:
                hard_neg_sims, _ = torch.topk(neg_matrix_for_topk, k=actual_k, dim=1)
                neg_scores = hard_neg_sims.mean(dim=1)
            else:
                neg_scores = torch.zeros_like(pos_scores)
        else:
            neg_scores = torch.zeros_like(pos_scores)

        # --- Part C: 差分 ---
        mcd_scores = pos_scores - neg_scores
        
        # --- DEBUG SECTION START ---
        # if torch.rand(1).item() < 0.1:
        #     print(f"[IADD DEBUG] PosScores Mean: {pos_scores.mean().item():.4f}")
        #     print(f"[IADD DEBUG] NegScores Mean: {neg_scores.mean().item():.4f}")
        #     print(f"[IADD DEBUG] MCD Mean: {mcd_scores.mean().item():.4f}")
        # --- DEBUG SECTION END ---

        return mcd_scores

    def get_hybrid_dist_matrix(self, dist_m1, dist_m2, weight_m1):
        W = weight_m1.view(-1, 1).expand_as(dist_m1)
        dist_hybrid = W * dist_m1 + (1 - W) * dist_m2
        return dist_hybrid

    def forward(self, logits_m1, logits_m2, feats_m1, feats_m2, labels):
        # 0. 特征归一化 (关键！如果不归一化，点积不是相似度)
        feats_m1_norm = F.normalize(feats_m1, p=2, dim=1)
        feats_m2_norm = F.normalize(feats_m2, p=2, dim=1)

        # 1. 计算 MCD
        mcd_m1 = self.compute_mcd_vectorized(feats_m1_norm, labels)
        mcd_m2 = self.compute_mcd_vectorized(feats_m2_norm, labels)
        
        # DEBUG: Ensure MCD is not 0
        if mcd_m1.abs().sum() < 1e-6:
             pass # Placeholder for breakpoint

        # 2. 动态权重
        # 此时 mcd 应该在 [-2, 2] 之间，通常在 [0, 1] 附近
        gap = mcd_m1 - mcd_m2
        alpha = self.sigmoid(gap * self.T)

        # 3. 蒸馏 Loss
        log_prob_m1 = F.log_softmax(logits_m1, dim=1)
        log_prob_m2 = F.log_softmax(logits_m2, dim=1)
        prob_m1 = F.softmax(logits_m1, dim=1)
        prob_m2 = F.softmax(logits_m2, dim=1)

        loss_m1_teach_m2 = self.kl_loss(log_prob_m2, prob_m1.detach()).sum(dim=1) 
        loss_m2_teach_m1 = self.kl_loss(log_prob_m1, prob_m2.detach()).sum(dim=1)

        loss_distill = (alpha * loss_m1_teach_m2 + (1 - alpha) * loss_m2_teach_m1).mean()

        # 4. 混合距离
        # 即使 feats 已经归一化，cdist 算的也是欧氏距离
        # ||x-y||^2 = 2 - 2cos(theta)
        dist_m1 = torch.cdist(feats_m1_norm, feats_m1_norm, p=2)
        dist_m2 = torch.cdist(feats_m2_norm, feats_m2_norm, p=2)
        
        hybrid_dist = self.get_hybrid_dist_matrix(dist_m1, dist_m2, alpha)

        return {
            "loss_distill": loss_distill * self.lambda_distill,
            "hybrid_dist": hybrid_dist,
            "mcd_m1": mcd_m1.mean().item(),
            "mcd_m2": mcd_m2.mean().item(),
            "alpha_mean": alpha.mean().item()
        }

class HybridTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(HybridTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, dist_mat, labels):
        n = dist_mat.size(0)
        labels = labels.view(n, 1)
        is_pos = torch.eq(labels, labels.t())
        
        # Hard Positive
        dist_ap, _ = torch.max(dist_mat * is_pos.float(), dim=1)
        
        # Hard Negative
        dist_mat_neg = dist_mat.clone()
        dist_mat_neg.masked_fill_(is_pos, float('inf'))
        dist_an, _ = torch.min(dist_mat_neg, dim=1)
        
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss
