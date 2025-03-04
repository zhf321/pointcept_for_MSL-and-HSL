import torch
import torch.nn as nn
import torch.nn.functional as F
class SupConLoss(nn.Module):

    def __init__(self, temperature=0.5, scale_by_temperature=True, is_rm_diag=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        self.is_rm_diag = is_rm_diag  # mask 是否移除正对对角线

    def forward(self, features=None, feats_sim=None, labels=None, mask=None, 
                weight=None, is_positive_in_denom=True, negatives_mask=None):
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            feats_sim: 特征相似度矩阵，如果提供feats_sim, 就不需要features了
            labels: 每个样本的ground truth标签，尺寸是[batch_size].
            mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 
                  如果样本i和j属于同一个label，那么mask_{i,j}=1 
            weight: 每一行计算 loss 的权重， shape 为 batch_size
            is_positive_in_denom: 在计算loss 时，是否要把 正对相似度列在分母中
            negative_mask: 若为None, 则~positive_mask即为负对，若提供负对，则按照
                           提供的负对mask 进行计算
        输出:
            loss值
        """
        if feats_sim is None:
            device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
            features = F.normalize(features, p=2, dim=1)
            batch_size = features.shape[0]

            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(features, features.T),
                self.temperature)  # 计算两两样本间点乘相似度
        else:
            device = feats_sim.device
            batch_size = feats_sim.shape[0]
            anchor_dot_contrast = feats_sim

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
        '''
        示例: 
        labels: 
            tensor([[1.],
                    [2.],
                    [1.],
                    [1.]])
        mask:  # 两个样本i,j的label相等时，mask_{i,j}=1
            tensor([[1., 0., 1., 1.],
                    [0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]]) 
        '''
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        '''
        logits是anchor_dot_contrast减去每一行的最大值得到的最终相似度
        示例: logits: torch.size([4,4])
        logits:
            tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
                    [-1.2576,  0.0000, -0.3367, -0.0725],
                    [-1.3500, -0.1409, -0.1420,  0.0000],
                    [-1.4312, -0.0776, -0.2009,  0.0000]])       
        '''
        # 构建mask 
        if self.is_rm_diag:
            logits_mask = torch.ones_like(mask) - torch.eye(batch_size)
        else:
            logits_mask = torch.ones_like(mask)     
        positives_mask = mask * logits_mask
        if negatives_mask is None:
            negatives_mask = 1. - mask
        else:
            # 如果某一行的负对 mask 全部为 False，则跳过该行的计算
            valid_negatives_mask = torch.any(negatives_mask, dim=1)
            positives_mask = positives_mask[valid_negatives_mask]
            negatives_mask = negatives_mask[valid_negatives_mask]
            logits = logits[valid_negatives_mask]
            exp_logits = exp_logits[valid_negatives_mask]
        
        '''
        但是对于计算Loss而言，(i,i)位置表示样本本身的相似度，
        对Loss是没用的，所以要mask掉
        # 第ind行第ind位置填充为0
        得到logits_mask:
            tensor([[0., 1., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 1., 0., 1.],
                    [1., 1., 1., 0.]])
        positives_mask:
        tensor([[0., 0., 1., 1.],
                [0., 0., 0., 0.],
                [1., 0., 0., 1.],
                [1., 0., 1., 0.]])
        negatives_mask:
        tensor([[0., 1., 0., 0.],
                [1., 0., 1., 1.],
                [0., 1., 0., 0.],
                [0., 1., 0., 0.]])
        '''        
        num_positives_per_row  = torch.sum(positives_mask , dim=1) # 除了自己之外，正样本的个数  [2 0 2 2] 

        if is_positive_in_denom:
            denominator = torch.sum(
                exp_logits * negatives_mask, dim=1, keepdims=True) + torch.sum(
                exp_logits * positives_mask, dim=1, keepdims=True)  
        else:
            denominator = torch.sum(
                exp_logits * negatives_mask, dim=1, keepdims=True)
            
        # 添加数值稳定性，防止分母为零
        epsilon = 1e-10
        denominator = denominator + epsilon
        
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        

        log_probs = torch.sum(
            log_probs*positives_mask , dim=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        
        if weight is not None:
            # print(f"log_probs.shape: {log_probs.shape}")
            # print(f"weight[num_positives_per_row > 0].shape: {weight[num_positives_per_row > 0].shape}")
            log_probs = log_probs * weight[num_positives_per_row > 0]

        '''
        计算正样本平均的log-likelihood
        考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        所以这里只计算正样本个数>0的    
        '''
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss