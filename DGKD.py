import torch
import torch.nn as nn
import torch.nn.functional as F
from corretation import difference
from visual import logits
def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-9 + stdv)
def dgkd_loss(logits_mlp, logits_gnn, target, alpha, beta, temperature):
    logits_mlp = normalize(logits_mlp)
    logits_gnn = normalize(logits_gnn)
    ##############

    #############
    # logits(logits_mlp,logits_gnn)
    # loss=rld_loss(logits_mlp, logits_gnn, target, alpha, beta, temperature, logit_stand=True, alpha_temperature=1.0)
    gt_mask = _get_gt_mask(logits_mlp, target)
    other_mask = _get_other_mask(logits_gnn, target)
    pred_mlp = F.softmax(logits_mlp / temperature, dim=1)
    pred_gnn = F.softmax(logits_gnn / temperature, dim=1)
    kd=(
        F.kl_div(pred_mlp, pred_gnn, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_mlp = cat_mask(pred_mlp, gt_mask, other_mask)
    pred_gnn = cat_mask(pred_gnn, gt_mask, other_mask)
    log_pred_mlp = torch.log(pred_mlp)
    tcgd_loss = (
        F.kl_div(log_pred_mlp, pred_gnn, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    pred_gnn_part2 = F.softmax(
        logits_gnn / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_mlp_part2 = F.log_softmax(
        logits_mlp / temperature - 1000.0 * gt_mask, dim=1
    )
    ncgd_loss = (
        F.kl_div(log_pred_mlp_part2, pred_gnn_part2, size_average=False)
        * (temperature**2)
        / target.shape[0]
    )
    nkd_loss=NKDLoss()(logits_mlp,logits_gnn,target)
    return  nkd_loss
    # return loss
    # return kd
    # return alpha * tcgd_loss + beta * ncgd_loss + 0.1 * nkd_loss
    # return alpha * tcgd_loss + beta * ncgd_loss


class NKDLoss(nn.Module):
    """ PyTorch version of NKD """

    def __init__(self,
                 temp=1.0,
                 gamma=1.5,
                 ):
        super(NKDLoss, self).__init__()

        self.temp = temp
        self.gamma = gamma
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logit_s, logit_t, gt_label):

        if len(gt_label.size()) > 1:
            label = torch.max(gt_label, dim=1, keepdim=True)[1]
        else:
            label = gt_label.view(len(gt_label), 1)

        # N*class
        N, c = logit_s.shape
        s_i = self.log_softmax(logit_s)
        t_i = F.softmax(logit_t, dim=1)
        # N*1
        s_t = torch.gather(s_i, 1, label)
        t_t = torch.gather(t_i, 1, label).detach()

        loss_t = - (t_t * s_t).mean()

        mask = torch.ones_like(logit_s).scatter_(1, label, 0).bool()
        logit_s = logit_s[mask].reshape(N, -1)
        logit_t = logit_t[mask].reshape(N, -1)

        # N*class
        S_i = self.log_softmax(logit_s / self.temp)
        T_i = F.softmax(logit_t / self.temp, dim=1)

        loss_non = (T_i * S_i).sum(dim=1).mean()
        loss_non = - self.gamma * (self.temp ** 2) * loss_non

        return loss_t + loss_non

def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt
import torch
import torch.nn as nn
import torch.nn.functional as F
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
