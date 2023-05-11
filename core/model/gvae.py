import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.append('.')
from core.functions.bayesianInfer import log_gaussian_prob, post_from_log_likelihood, post_from_likelihood, guassian_kl, discrete_kl

class GMMVAE(nn.Module): 
    def __init__(self, feat_dim=1024, embed_dim=128, n_cluster=4, n_class=20, 
                 use_learnable_mix_ratio=False, approx_prior=False, wta=False):
        super().__init__()
        self.embed_dim = embed_dim
        print(f'init gmm-vae with n_class={n_class}, n_cluster={n_cluster}')
        self.n_cluster = n_cluster
        self.n_class = n_class

        self.mean = nn.Parameter(torch.randn(n_class*2, n_cluster, embed_dim))
        self.log_var = nn.Parameter(torch.zeros(n_class*2, n_cluster, embed_dim))

        self.wta = wta
        self.use_learnable_mix_ratio = use_learnable_mix_ratio
        if use_learnable_mix_ratio == True:
            self.mix_ratio = nn.Parameter(torch.ones(n_class*2, n_cluster) / n_cluster) # should keep normlized
        else:
            self.register_buffer('mix_ratio', torch.ones(n_class*2, n_cluster) / n_cluster)

        self.approx_prior = approx_prior

        self.relu = nn.ReLU()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            self.relu,
            nn.Linear(embed_dim, embed_dim),
            self.relu,
            nn.Linear(embed_dim, 2*embed_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            self.relu,
            nn.Linear(embed_dim, embed_dim),
            self.relu,
            nn.Linear(embed_dim, feat_dim)
        )
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        for m in self.decoder:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def _sum_norm_ratio(self):
        self.mix_ratio = self.mix_ratio / self.mix_ratio.sum(dim=-1, keepdim=True)

    def log_cls_likelihood_from_log_cluster_likelihood(self, log_likelihood, wta=False):
        if wta:
            log_likelihood = log_likelihood.max(dim=-1)[0] #[seq_len, C]
        else:
            scale1 = 88 - log_likelihood.max(dim=-1,keepdim=True)[0] #[seq_len, C, 1]
            scaled_likelihood = (torch.exp(log_likelihood+scale1)*self.mix_ratio.unsqueeze(0)) #[seq_len, C, K]
            scaled_likelihood = scaled_likelihood.sum(dim=2) #[seq_len, C]
            log_likelihood = torch.log(scaled_likelihood) - scale1.squeeze(2) #[seq_len, C]
        return log_likelihood

    def infer_gmm(self, z_l, prior):
        log_p_zc_list = []
        for i in range(self.n_class * 2):
            log_p_zc_list.append(log_gaussian_prob(z_l.unsqueeze(1), 
                                                   self.mean[i].unsqueeze(0), self.log_var[i].unsqueeze(0))) #[bs, K]
        log_p_zc = torch.stack(log_p_zc_list, dim=1)
        p_cz = post_from_log_likelihood(log_p_zc, self.mix_ratio.unsqueeze(0)) #[bs, C, K]
        log_p_zlam = self.log_cls_likelihood_from_log_cluster_likelihood(log_p_zc, self.wta)
        # print(f'nan in log_p_zlam = {(torch.isnan(log_p_zlam).any())}')
        # p_lamz = post_from_log_likelihood(log_p_zlam, prior) #[bs, C]
        p_lamz = post_from_likelihood(log_p_zlam, prior) #[bs, C]
        # print(f'nan in p_lamz = {(torch.isnan(p_lamz)).any()}')
        return p_cz, p_lamz, log_p_zlam

    def kl_inner(self, z_mean, z_log_var, gmm_post, cls_prob):
        # q(z|x) p(z|c)
        kl_inner1 = guassian_kl(z_mean.unsqueeze(1).unsqueeze(1), z_log_var.unsqueeze(1).unsqueeze(1),
                                 self.mean.unsqueeze(0), self.log_var.unsqueeze(0)) #[bs, C, K]
        # print(f'kl_inner1={kl_inner1}')
        # p(c|z) p(c|lambda)
        kl_inner2 = discrete_kl(gmm_post, self.mix_ratio.unsqueeze(0))#[bs, C]
        # print(f'kl_inner2={kl_inner2}')
        # print(cls_prob)
        # print(f'mix_ratio={self.mix_ratio}')
        kl_inner = (kl_inner1 * gmm_post).sum(dim=-1) + kl_inner2 #[bs, C]
        kl_inner = (kl_inner * cls_prob).sum(dim=-1) #[bs]
        return kl_inner.mean()

    def kl_prior(self, prior, cls_prob):
        if self.approx_prior:
            kl_prior = discrete_kl(cls_prob, prior) #[bs]
            kl_prior = kl_prior.mean()
        else:
            kl_prior = torch.scalar_tensor(0, device=prior.device)
        return kl_prior

    def forward_gmm(self, z_l, z_mean, z_log_var, prior):
        self._sum_norm_ratio()
        # calculate p(c|z), p(lambda|z)
        p_cz, p_lamz, log_p_zlam = self.infer_gmm(z_l, prior)
    
        # calculate kl_inner using prior or p(lambda|z)
        cls_prob = p_lamz if self.approx_prior else prior
        kl_inner = self.kl_inner(z_mean, z_log_var, p_cz, cls_prob)

        # calculate kl_prior 0 or p(lambda|z) prior
        kl_prior = self.kl_prior(prior, p_lamz)
        return kl_inner, kl_prior, p_lamz, log_p_zlam

    def forward(self, x, prior):
        bs, dim, h, w = x.shape
        x = x.permute(0,2,3,1).flatten(0,2) #[seq_len, dim]
        prior=  prior.permute(0,2,3,1).flatten(0,2) #[seq_len, 2C]

        z = self.encoder(x) #[seq_len, 2*dim]

        mean = z[:, :self.embed_dim] #[seq_len, dim]
        log_var = z[:, self.embed_dim:] #[seq_len, dim]
        std = torch.exp(0.5 * log_var) #[seq_len, dim]
        eps = torch.randn_like(std) #[seq_len, dim]
        z_l = mean + eps * std #[seq_len, dim]

        recon = self.decoder(z_l)

        kl_inner, kl_prior, p_lamz, log_p_zlam = self.forward_gmm(z_l, mean, log_var, prior)
        recon_loss = F.mse_loss(recon, x, reduction='none').sum(dim=-1).mean()

        return recon_loss, kl_inner, kl_prior, log_p_zlam.reshape(bs, h, w, -1).permute(0, 3, 1, 2),\
            p_lamz.reshape(bs, h, w, -1).permute(0, 3, 1, 2)


if __name__ == "__main__":
    from core.misc.utils import setup_seed
    setup_seed(0)

    gvae = GMMVAE(feat_dim=16, embed_dim=8, n_cluster=2, n_class=3,
                   use_learnable_mix_ratio=False, approx_prior=True, wta=False)


    x = torch.randn(2, 16, 4, 4) #[bs, dim, h, w]
    prior = F.normalize(torch.ones(2, 6, 4, 4)).abs() #[bs, 2C, h, w]
    prior = prior / prior.sum(dim=1,keepdim=True)


    recon_loss, kl_inner, kl_prior, log_p_zlam, p_lamz = gvae(x, prior)
    print(f'recon_loss={recon_loss}')
    print(f'kl_inner={kl_inner}')
    print(f'kl_prior={kl_prior}')