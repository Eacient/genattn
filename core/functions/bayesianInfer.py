import math
import torch
import torch.nn.functional as F
 
def log_gaussian_prob(z, mu, log_var):
    # print(f'gaussian shapes{z.shape} {mu.shape} {log_var.shape}')
    log_prob = -(z-mu)**2/(2*torch.exp(log_var)) - math.log(math.sqrt(2*math.pi)) - 0.5*log_var #[seq_len, c, k, dim]
    log_prob = torch.sum(log_prob, dim=-1) #[seq_len, c, k]
    # print(f'log_prob < 0={torch.sum(log_prob==0).item()}')
    return log_prob
 
def guassian_kl(mu_1, logvar_1, mu_2, logvar_2):
    J = mu_1.shape[-1]
    return -J/2 + ((logvar_2 - logvar_1) + torch.exp(logvar_1-logvar_2) + 
                 (mu_1 - mu_2)**2/torch.exp(logvar_2)).sum(dim=-1)/2

def discrete_kl(q, p):
    # print(f'cluster_post zeros {torch.sum(q==0)}')
    # print(f'mix_ratio zeros {torch.sum(p==0)}')
    q = q + 1e-8;q = q / (q.sum(dim=-1, keepdim=True))
    p = p + 1e-8;p = p / (p.sum(dim=-1, keepdim=True))
    dkl_term = torch.sum(q * torch.log(q/p), dim=-1) #[seq_len, C]
    # print(f'nan in dkl={torch.isnan(dkl_term).any()}')
    return dkl_term

def post_from_log_likelihood(log_likelihood, prior):
    log_joint = log_likelihood + prior.log() #[seq_len, ?, K]
    log_joint = log_joint + 88 - torch.max(log_joint, dim=-1, keepdim=True)[0] #[seq_len, ?, K]
    post = F.softmax(log_joint, dim=-1) #[seq_len, ?, K]
    return post

def post_from_likelihood(log_likelihood, prior):
    shifted_likelihood = torch.exp(log_likelihood - torch.max(log_likelihood, dim=-1, keepdim=True)[0])
    joint = shifted_likelihood * prior
    post = joint / (joint.sum(dim=-1, keepdim=True) + 1e-8)
    return post
