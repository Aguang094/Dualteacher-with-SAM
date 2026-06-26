import torch
import torch.nn as nn
import torch.nn.functional as F

def softmax_entropy(logits: torch.Tensor, dim: int = 1, eps: float = 1e-10) -> torch.Tensor:
    """
    计算 Softmax 熵
    logits: [B, C, D, H, W] (or [B, C, H, W])
    return: entropy [B, D, H, W] (or [B, H, W])
    [cite: 1]
    """
    probs = F.softmax(logits, dim=dim)
    log_probs = torch.log(probs + eps)
    return -(probs * log_probs).sum(dim=dim)


def top1_top2_gap(probs: torch.Tensor) -> torch.Tensor:
    """
    计算 Top1 和 Top2 概率之差 (置信度 Gap)
    probs: [B, C, ...]
    return: gap [B, ...] = p1 - p2
    [cite: 2]
    """
    top2 = torch.topk(probs, k=2, dim=1, largest=True, sorted=True).values
    return top2[:, 0] - top2[:, 1]


def erode_mask_3d(binary_mask: torch.Tensor, k: int = 3, iters: int = 1) -> torch.Tensor:
    """
    近似 3D 形态学腐蚀：erode(x) = 1 - dilate(1-x)
    binary_mask: [B, 1, D, H, W] in {0,1}
    [cite: 3]
    """
    assert binary_mask.dim() == 5, "binary_mask must be [B,1,D,H,W]"
    x = binary_mask.float()
    pad = k // 2
    for _ in range(iters):
        inv = 1.0 - x
        dil = F.max_pool3d(inv, kernel_size=k, stride=1, padding=pad)
        x = 1.0 - dil
    return x


class BoundaryContrastiveLoss(nn.Module):
    """
    Student-only 版 Boundary Contrastive / Prototype Calibration Loss（支持3D）
    包含 Local Prototypes (Pixel-based) 和 Global Prototypes (Query-based) 的对比。
    [cite: 4, 5, 6]
    """

    def __init__(
        self,
        temperature: float = 0.07,

        # prototypes 选择：类内 low-entropy 百分比 + 每类最大点数
        low_percent: float = 0.10,
        max_proto_features_per_class: int = 500,

        # anchors 选择：整体 high-entropy 百分比（从剩余里选）+ 剔除极端高熵比例 + 最大anchor数
        high_percent: float = 0.10,
        discard_percent: float = 0.10,
        max_anchors_per_batch: int = 2048,

        # 是否从 GT 内部区域取 prototype（避免 GT 边界噪声）
        use_gt_interior_for_proto: bool = True,
        interior_erode_iters: int = 1,
        interior_kernel_size: int = 3,

        # anchor gate：避免完全随机/噪声点（建议开着）
        min_pmax_for_anchor: float = 0.20,

        # 可选：对极高置信 anchor 再加一个 hard CE（更锐）
        use_hard_ce: bool = True,
        hard_pmax: float = 0.85,
        hard_gap: float = 0.20,

        # 损失权重
        lambda_soft: float = 1.0,   # Local Prototypes 权重
        lambda_global: float = 0.5, # Global (Query) Prototypes 权重 [cite: 48]
        lambda_hard: float = 1.0,

        eps: float = 1e-10,
        
        # [NEW] 维度对齐参数
        feature_dim: int = 16,  # UNet输出的特征维度 (rep_u的通道数)
        query_dim: int = 64,    # kMaX Query的维度
    ):
        super().__init__()
        self.temperature = temperature

        self.low_percent = low_percent
        self.max_proto_features_per_class = max_proto_features_per_class

        self.high_percent = high_percent
        self.discard_percent = discard_percent
        self.max_anchors_per_batch = max_anchors_per_batch

        self.use_gt_interior_for_proto = use_gt_interior_for_proto
        self.interior_erode_iters = interior_erode_iters
        self.interior_kernel_size = interior_kernel_size

        self.min_pmax_for_anchor = min_pmax_for_anchor

        self.use_hard_ce = use_hard_ce
        self.hard_pmax = hard_pmax
        self.hard_gap = hard_gap

        self.lambda_soft = lambda_soft
        self.lambda_global = lambda_global
        self.lambda_hard = lambda_hard

        self.eps = eps

        # [NEW] 线性投影层：将 kMaX Query 映射到 UNet Feature 空间
        # 解决  中提到的维度不匹配问题
        if lambda_global > 0 and feature_dim != query_dim:
            self.query_proj = nn.Linear(query_dim, feature_dim, bias=False)
            # 初始化投影层
            nn.init.orthogonal_(self.query_proj.weight)
        else:
            self.query_proj = nn.Identity()

    @staticmethod
    def _flatten_features(feat: torch.Tensor) -> torch.Tensor:
        # [B, F, D, H, W] -> [B, N, F]
        b, f = feat.shape[0], feat.shape[1]
        return feat.view(b, f, -1).permute(0, 2, 1).contiguous()

    @staticmethod
    def _flatten_spatial(x: torch.Tensor) -> torch.Tensor:
        # [B, ...] -> [B, N]
        return x.view(x.shape[0], -1)

    def _build_prototypes_from_labeled(
        self,
        seg_l: torch.Tensor,     # [B,C,D,H,W]
        feat_l: torch.Tensor,    # [B,F,D,H,W]
        y_l: torch.Tensor,       # [B,D,H,W]
    ):
        """
        构建局部原型 (Local Prototypes)
        return:
          protos: [B, C, F]
          proto_valid: [B, C] bool
        [cite: 12, 13]
        """
        device = seg_l.device
        B, C = seg_l.shape[0], seg_l.shape[1]
        Fdim = feat_l.shape[1]

        probs_l = F.softmax(seg_l, dim=1)
        ent_l = softmax_entropy(seg_l, dim=1, eps=self.eps)  # [B,D,H,W]

        feats_flat = self._flatten_features(feat_l)  # [B,N,F]
        ent_flat = self._flatten_spatial(ent_l)      # [B,N]
        y_flat = y_l.view(B, -1).long()              # [B,N]

        # 内部区域 mask（可选）
        if self.use_gt_interior_for_proto:
            # 对每个类做腐蚀后求 interior
            interior_masks = torch.zeros((B, C, *y_l.shape[1:]), device=device, dtype=torch.float32)
            for c in range(C):
                m = (y_l == c).float().unsqueeze(1)  # [B,1,D,H,W]
                m_int = erode_mask_3d(m, k=self.interior_kernel_size, iters=self.interior_erode_iters)
                interior_masks[:, c] = m_int.squeeze(1)
            interior_flat = interior_masks.view(B, C, -1)  # [B,C,N]
        else:
            interior_flat = None

        protos = torch.zeros((B, C, Fdim), device=device)
        proto_valid = torch.zeros((B, C), device=device, dtype=torch.bool)

        for b in range(B):
            for c in range(C):
                cls_mask = (y_flat[b] == c)
                if interior_flat is not None:
                    cls_mask = cls_mask & (interior_flat[b, c] > 0.5)

                if not torch.any(cls_mask):
                    continue

                ent_c = ent_flat[b][cls_mask]
                n = ent_c.numel()
                k = max(1, int(self.low_percent * n))
                k = min(k, self.max_proto_features_per_class)

                # 选 entropy 最小的 k 个（用 topk largest=False，避免全排序）
                _, idx_local = torch.topk(ent_c, k=k, largest=False, sorted=True)
                idx_global = torch.nonzero(cls_mask, as_tuple=True)[0][idx_local]

                feats_c = feats_flat[b][idx_global]  # [k,F]
                protos[b, c] = feats_c.mean(dim=0)
                proto_valid[b, c] = True

        return protos, proto_valid

    def _select_unlabeled_anchors(
        self,
        seg_u: torch.Tensor,     # [B,C,D,H,W]
        feat_u: torch.Tensor,    # [B,F,D,H,W]
    ):
        """
        选 anchor：高熵，但剔除最极端一段；并加 min_pmax gate 避免纯噪声点
        return per-batch:
          anchors_feat: list[Tensor] each [Nu,F]
          anchors_prob: list[Tensor] each [Nu,C] (student probs at anchor positions, for soft label)
          anchors_hard_mask: list[Tensor] each [Nu] bool  (for optional hard CE)
          anchors_hard_y: list[Tensor] each [Nu] long     (argmax class)
        [cite: 20]
        """
        device = seg_u.device
        B, C = seg_u.shape[0], seg_u.shape[1]
        Fdim = feat_u.shape[1]

        probs_u = F.softmax(seg_u, dim=1)                    # [B,C,D,H,W]
        ent_u = softmax_entropy(seg_u, dim=1, eps=self.eps)  # [B,D,H,W]
        pmax_u, y_hat_u = probs_u.max(dim=1)                 # [B,D,H,W]
        gap_u = top1_top2_gap(probs_u)                       # [B,D,H,W]

        feats_flat = self._flatten_features(feat_u)          # [B,N,F]
        ent_flat = self._flatten_spatial(ent_u)              # [B,N]
        pmax_flat = self._flatten_spatial(pmax_u)            # [B,N]
        gap_flat = self._flatten_spatial(gap_u)              # [B,N]
        probs_flat = probs_u.view(B, C, -1).permute(0, 2, 1).contiguous()  # [B,N,C]
        yhat_flat = y_hat_u.view(B, -1).long()               # [B,N]

        anchors_feat, anchors_prob = [], []
        anchors_hard_mask, anchors_hard_y = [], []

        for b in range(B):
            # gate：避免 pmax 太低的纯噪声点
            valid = (pmax_flat[b] >= self.min_pmax_for_anchor)
            if not torch.any(valid):
                anchors_feat.append(torch.zeros((0, Fdim), device=device))
                anchors_prob.append(torch.zeros((0, C), device=device))
                anchors_hard_mask.append(torch.zeros((0,), device=device, dtype=torch.bool))
                anchors_hard_y.append(torch.zeros((0,), device=device, dtype=torch.long))
                continue

            ent_v = ent_flat[b][valid]
            idx_v = torch.nonzero(valid, as_tuple=True)[0]  # global indices for valid

            n = ent_v.numel()
            if n <= 1:
                anchors_feat.append(feats_flat[b][idx_v[:1]])
                anchors_prob.append(probs_flat[b][idx_v[:1]])
                anchors_hard_mask.append(torch.zeros((1,), device=device, dtype=torch.bool))
                anchors_hard_y.append(yhat_flat[b][idx_v[:1]])
                continue

            # 剔除最极端高熵 discard_percent
            discard_num = int(self.discard_percent * n)
            discard_num = max(0, min(discard_num, n - 1))

            # 从剩余里选 high_percent
            remain_num = n - discard_num
            k = max(1, int(self.high_percent * remain_num))
            k = min(k, self.max_anchors_per_batch)

            # 取 top (discard+k) 的高熵，然后丢掉最前 discard
            take = min(n, discard_num + k)
            _, idx_take = torch.topk(ent_v, k=take, largest=True, sorted=True)
            idx_sel_local = idx_take[discard_num:discard_num + k]
            idx_sel_global = idx_v[idx_sel_local]

            a_feat = feats_flat[b][idx_sel_global]   # [k,F]
            a_prob = probs_flat[b][idx_sel_global]   # [k,C]
            a_y = yhat_flat[b][idx_sel_global]       # [k]

            # hard CE mask（极高置信才用）
            if self.use_hard_ce:
                a_pmax = pmax_flat[b][idx_sel_global]
                a_gap = gap_flat[b][idx_sel_global]
                hard_mask = (a_pmax >= self.hard_pmax) & (a_gap >= self.hard_gap)
            else:
                hard_mask = torch.zeros((a_feat.shape[0],), device=device, dtype=torch.bool)

            anchors_feat.append(a_feat)
            anchors_prob.append(a_prob)
            anchors_hard_mask.append(hard_mask)
            anchors_hard_y.append(a_y)

        return anchors_feat, anchors_prob, anchors_hard_mask, anchors_hard_y

    def _proto_logits(self, anchor_feat: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
        """
        anchor_feat: [N,F]
        protos: [V,F]
        return logits: [N,V] = cos/temperature
        [cite: 33]
        """
        a = F.normalize(anchor_feat, p=2, dim=1)
        p = F.normalize(protos, p=2, dim=1)
        return (a @ p.t()) / max(self.temperature, self.eps)

    # =========================================================
    # [NEW] 核心改动：从 Query 构建全局原型，并添加了 Projection
    # =========================================================
    def _build_global_prototypes_from_queries(
        self, 
        current_query: torch.Tensor,   # [B, Qdim, Num_Queries]
        class_logits: torch.Tensor,    # [B, Num_Queries, Num_Classes]
        num_classes: int
    ):
        """
        将 kMaX 的 Query 聚合为 C 个 Global Prototypes。
        聚合方式：加权平均。Query 对某个 Class 的预测概率越高，它对该 Class 原型的贡献越大。
        [cite: 67, 68]
        """
        B, Qdim, NQ = current_query.shape
        
        # 1. 准备 Query 特征并投影
        q_feat = current_query.permute(0, 2, 1) # [B, NQ, Qdim]
        
        # [FIX] 投影到 Feature Space, 解决  的维度问题
        q_feat = self.query_proj(q_feat) # [B, NQ, Fdim]

        # 2. 获取权重: [B, NQ, C]
        # 使用 Softmax 归一化 Query 对 Class 的贡献
        query_class_weights = F.softmax(class_logits, dim=-1) # [B, NQ, C+1] if void exists
        
        # 只取前 num_classes 个前景类
        query_class_weights = query_class_weights[..., :num_classes] # [B, NQ, C]

        # 3. 加权聚合 (Einsum): 
        # w: [B, NQ, C], q: [B, NQ, Fdim] -> global_protos: [B, C, Fdim]
        # 公式: sum_{NQ} (Weight_{NQ, C} * Query_{NQ, Fdim})
        global_protos = torch.einsum('bnc,bnd->bcd', query_class_weights, q_feat)
        
        # 4. 归一化 (除以权重的和，保证尺度)
        weight_sum = query_class_weights.sum(dim=1, keepdim=True).transpose(1, 2) # [B, C, 1]
        global_protos = global_protos / (weight_sum + self.eps)
        
        # 5. 生成 Valid Mask
        global_valid = (weight_sum.squeeze(-1) > 1e-4) # [B, C]
        
        return global_protos, global_valid

    def forward(
        self,
        seg_l: torch.Tensor, feat_l: torch.Tensor, y_l: torch.Tensor,
        seg_u: torch.Tensor, feat_u: torch.Tensor,
        # [NEW Inputs]
        current_query: torch.Tensor = None, # [B, Qdim, Num_Queries]
        kmax_class_logits: torch.Tensor = None # [B, Num_Queries, Num_Classes]
    ) -> torch.Tensor:
        """
        Forward Pass
        seg_l/seg_u: logits [B,C,D,H,W]
        feat_l/feat_u: [B,F,D,H,W]
        y_l: [B,D,H,W]
        """
        device = seg_l.device
        B, C = seg_l.shape[0], seg_l.shape[1]

        # 1) [Local] Prototypes from labeled pixels
        protos_local, valid_local = self._build_prototypes_from_labeled(seg_l, feat_l, y_l)

        # 2) [Global] Prototypes from kMaX queries (If provided)
        use_global = (current_query is not None) and (kmax_class_logits is not None) and (self.lambda_global > 0)
        if use_global:
            protos_global, valid_global = self._build_global_prototypes_from_queries(
                current_query, kmax_class_logits, num_classes=C
            )

        # 3) Anchors from unlabeled pixels
        anchors_feat, anchors_prob, anchors_hard_mask, anchors_hard_y = self._select_unlabeled_anchors(seg_u, feat_u)

        total_loss = torch.tensor(0.0, device=device)
        total_count = 0

        for b in range(B):
            a_feat = anchors_feat[b] # [N, F]
            if a_feat.numel() == 0:
                continue

            # --- A. Local Contrast (Anchor vs Pixel Prototypes) ---
            loss_b_local = torch.tensor(0.0, device=device)
            valid_idx_local = torch.nonzero(valid_local[b], as_tuple=True)[0]
            
            if valid_idx_local.numel() >= 2:
                p_local = protos_local[b][valid_idx_local] # [V, F]
                
                # Align probabilities
                a_prob_full = anchors_prob[b] # [N, C]
                a_prob_loc = a_prob_full[:, valid_idx_local]
                a_prob_loc = a_prob_loc / (a_prob_loc.sum(dim=1, keepdim=True) + self.eps)
                
                # Sim & KL
                logits_loc = self._proto_logits(a_feat, p_local)
                log_q_loc = F.log_softmax(logits_loc, dim=1)
                loss_b_local = F.kl_div(log_q_loc, a_prob_loc, reduction="batchmean")
                
                # Hard CE (Optional)
                if self.use_hard_ce and torch.any(anchors_hard_mask[b]):
                    # hard label in local index of valid_classes
                    hard_mask = anchors_hard_mask[b]
                    y_hard_global = anchors_hard_y[b][hard_mask]  # [Nh]
                    
                    # map global class id -> local
                    cls_to_local = {int(c.item()): i for i, c in enumerate(valid_idx_local)}
                    local_indices = []
                    keep_indices = []
                    
                    for i in range(y_hard_global.numel()):
                        g = int(y_hard_global[i].item())
                        if g in cls_to_local:
                            local_indices.append(cls_to_local[g])
                            keep_indices.append(i)
                    
                    if len(keep_indices) > 0:
                        keep_indices = torch.tensor(keep_indices, device=device, dtype=torch.long)
                        local_indices = torch.tensor(local_indices, device=device, dtype=torch.long)
                        loss_hard = F.cross_entropy(logits_loc[hard_mask][keep_indices], local_indices)
                        loss_b_local = loss_b_local + self.lambda_hard * loss_hard

            # --- B. [NEW] Global Contrast (Anchor vs Query Prototypes) ---
            loss_b_global = torch.tensor(0.0, device=device)
            if use_global:
                valid_idx_global = torch.nonzero(valid_global[b], as_tuple=True)[0]
                if valid_idx_global.numel() >= 2:
                    p_global = protos_global[b][valid_idx_global] # [V, F] (After projection)
                    
                    a_prob_full = anchors_prob[b]
                    a_prob_glb = a_prob_full[:, valid_idx_global]
                    a_prob_glb = a_prob_glb / (a_prob_glb.sum(dim=1, keepdim=True) + self.eps)
                    
                    logits_glb = self._proto_logits(a_feat, p_global)
                    log_q_glb = F.log_softmax(logits_glb, dim=1)
                    
                    loss_b_global = F.kl_div(log_q_glb, a_prob_glb, reduction="batchmean")

            # Combine Losses
            loss = 0.0
            if valid_idx_local.numel() >= 2:
                loss += self.lambda_soft * loss_b_local
            
            if use_global and valid_idx_global.numel() >= 2:
                loss += self.lambda_global * loss_b_global
                
            total_loss = total_loss + loss
            total_count += 1

        if total_count == 0:
            return total_loss * 0.0

        return total_loss / total_count