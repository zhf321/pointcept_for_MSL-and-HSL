import torch
import torch.nn as nn
import torch.nn.functional as F
class SupConLoss(nn.Module):

    def __init__(self, temperature=0.5, scale_by_temperature=True, is_rm_diag=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        self.is_rm_diag = is_rm_diag  # mask 是否移除正对对角线

    def forward(self, features, labels=None, mask=None):
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            labels: 每个样本的ground truth标签，尺寸是[batch_size].
            mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 
                  如果样本i和j属于同一个label，那么mask_{i,j}=1 
        输出:
            loss值
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # 关于labels参数
        if labels is not None and mask is not None: 
            # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的 
            raise ValueError('Cannot define both `labels` and `mask`') 
        elif labels is None and mask is None: 
            # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None: 
            # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        # 构建mask 
        if self.is_rm_diag:
            logits_mask = torch.ones_like(mask) - torch.eye(batch_size)
        else:
            logits_mask = torch.ones_like(mask)     
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
   
        num_positives_per_row  = torch.sum(positives_mask , dim=1) # 除了自己之外，正样本的个数  [2 0 2 2]       
        denominator = torch.sum(
        exp_logits * negatives_mask, dim=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, dim=1, keepdims=True)  
        
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        

        log_probs = torch.sum(
            log_probs*positives_mask , dim=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]

        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss