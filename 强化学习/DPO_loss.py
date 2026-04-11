import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    直接偏好优化损失 (Direct Preference Optimization Loss)
    DPO 是一种不使用奖励模型的 RLHF 替代方案，直接在人类偏好数据上优化语言模型。
    公式: L_DPO = -E[ log σ( β * (log(π_θ(y_w|x)/π_ref(y_w|x)) - log(π_θ(y_l|x)/π_ref(y_l|x))) ) ]
    """
    # ---------- 步骤1: 计算隐式奖励（对数比率） ----------
    chosen_ratio = policy_chosen_logps - ref_chosen_logps      # log(π_θ(y_w)/π_ref(y_w))
    rejected_ratio = policy_rejected_logps - ref_rejected_logps  # log(π_θ(y_l)/π_ref(y_l))

    # ---------- 步骤2: 计算 DPO logits ----------
    logits = chosen_ratio - rejected_ratio                     # 奖励差值

    # ---------- 步骤3: 计算 DPO 损失 ----------
    loss = -F.logsigmoid(beta * logits).mean()

    # ---------- 步骤4: 可选标签平滑 ----------
    if label_smoothing > 0.0:
        inverse_loss = -F.logsigmoid(-beta * logits).mean()
        loss = (1 - label_smoothing) * loss + label_smoothing * inverse_loss

    return loss
