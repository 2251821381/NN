import torch
import torch.nn as nn
import torch.nn.functional as F
from corretation import difference
from visual import logits


def rld_loss(logits_student_in, logits_teacher_in, target, alpha, beta, temperature, logit_stand, alpha_temperature):
    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    # scd loss
    student_gt_mask = _get_gt_mask(logits_student, target)
    student_other_mask = _get_other_mask(logits_student, target)
    max_index = torch.argmax(logits_teacher, dim=1)
    teacher_max_mask = _get_gt_mask(logits_teacher, max_index)
    teacher_other_mask = _get_other_mask(logits_teacher, max_index)
    pred_student = F.softmax(logits_student / alpha_temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / alpha_temperature, dim=1)
    pred_student = cat_mask(pred_student, student_gt_mask, student_other_mask)
    pred_teacher = cat_mask(pred_teacher, teacher_max_mask, teacher_other_mask)
    log_pred_student = torch.log(pred_student)
    scd_loss = F.kl_div(log_pred_student, pred_teacher, reduction='batchmean') * (alpha_temperature**2)

    # mcd loss
    mask = _get_ge_mask(logits_teacher, target)
    assert mask.shape == logits_student.shape
    masked_student = (logits_student / temperature).masked_fill(mask, -1e9)
    log_pred_student_part2 = F.log_softmax(masked_student, dim=1)
    masked_teacher = (logits_teacher / temperature).masked_fill(mask, -1e9)
    pred_teacher_part2 = F.softmax(masked_teacher, dim=1)
    mcd_loss = F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='batchmean') * (temperature**2)

    return alpha * scd_loss + beta * mcd_loss


def _get_ge_mask(logits, target):
    assert logits.dim() == 2 and target.dim() == 1 and logits.size(0) == target.size(0)
    gt_value = torch.gather(logits, 1, target.unsqueeze(1))
    mask = torch.where(logits >= gt_value, 1, 0).bool()
    return mask


def get_ratio(teacher_logits, logits, mu=0.5):
    # 输入：teacher_logits 和 logits 是 [batch, vocab] 张量
    teacher_logits = torch.masked_fill(teacher_logits, torch.isinf(teacher_logits), 0).to(torch.float32)
    logits = torch.masked_fill(logits, torch.isinf(logits), 0).to(torch.float32)
    # 处理无穷大值并转换为 float32 类型

    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32).detach()
    # 计算概率分布

    re_teacher_probs, idx = teacher_probs.sort(dim=-1, descending=True)
    re_student_probs = student_probs.gather(dim=-1, index=idx)
    # 对 teacher_probs 进行降序排序，并用相同顺序对 student_probs 排序

    errors = torch.abs(re_teacher_probs - re_student_probs)
    # 计算排序后的 teacher_probs 和 student_probs 之间的绝对误差

    cum_sum = torch.cumsum(re_teacher_probs, dim=-1)
    # 计算排序后的 teacher_probs 的累积和

    mask = cum_sum > mu
    mask[:, 0] = False
    # 创建掩码，将第一个概率的位置设为未掩码状态

    s1 = torch.masked_fill(errors, mask, 0.0).sum(dim=-1)
    s2 = torch.masked_fill(errors, ~mask, 0.0).sum(dim=-1)
    # 计算高重要性部分（s1）和低重要性部分（s2）的误差和

    return s1 / (s1 + s2), s2 / (s1 + s2)
    # 返回归一化的高重要性和低重要性部分的比例


def get_kl(teacher_logits, logits, inf_mask, mask, ratio=None):
    # 输入：teacher_logits 和 logits 是 [batch, vocab]，inf_mask 是 [batch, vocab] 掩码，mask 是 [batch] 掩码

    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    teacher_prod_probs = torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    teacher_x = torch.sum(teacher_prod_probs, dim=-1)
    # 计算教师概率和对数概率的乘积，并对无效位置进行屏蔽

    logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1)
    # 计算教师概率和学生对数概率的乘积

    if ratio is None:
        distil_loss = torch.sum((teacher_x - x) * mask, dim=0) / torch.sum(mask, dim=0)
    else:
        distil_loss = torch.sum((teacher_x - x) * ratio * mask, dim=0) / torch.sum(mask, dim=0)
    # 根据 ratio 是否存在来计算蒸馏损失，并应用掩码

    return distil_loss
    # 返回蒸馏损失


def AKL(teacher_logits, logits,label):
    # 输入：teacher_logits 和 logits 是 [batch, vocab] 张量

    inf_mask = torch.isinf(logits)
    # 为 logits 中的无穷大值创建掩码

    mask = (label != -100).int()
    # 创建掩码，标记有效的批次位置

    h_ratio, l_ratio = get_ratio(teacher_logits, logits)
    # 计算高重要性和低重要性比例

    distil_loss = get_kl(teacher_logits, logits, inf_mask, mask, h_ratio) + get_kl(logits, teacher_logits, inf_mask,
                                                                                   mask, l_ratio)
    # 使用高重要性和低重要性比例来计算总的蒸馏损失

    return distil_loss
    # 返回计算得到的自适应知识蒸馏损失
