from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from model_bak import gvae_cam
from dataset.VOC12Dataset import VOC12Dataset
import numpy as np
import torch
import torch.nn.functional as F




def prepare_numpy(model, dataset, indexes):
    conv5s = []
    zmeans = []
    segs = []
    cls_labels = []
    # forward hook for conv5 and z_mean
    def collect_orig_feat_and_reduced_feat(module, input, output):
        # print(input[0].shape)
        # print(output.shape)
        conv5s.append(input[0].detach().cpu().numpy())
        zmeans.append(output[:, :256].detach().cpu().numpy())
    ### using some forward hook
    model.to(torch.device('cuda'))
    model.gvae.encoder.register_forward_hook(collect_orig_feat_and_reduced_feat)
    for i in indexes:
        img_id, img, label, sal, seg, sp = dataset[i]
        img = img.unsqueeze(0)
        label = label.unsqueeze(0)
        sal = sal.unsqueeze(0)
        sp = sp.unsqueeze(0)
        seg = seg.unsqueeze(0)
        img = img.to(torch.device('cuda'))
        label = label.to(torch.device('cuda'))
        sal = sal.to(torch.device('cuda')).unsqueeze(1)
        sp = sp.to(torch.device('cuda')).unsqueeze(1)
        cls_loss, seg_loss, wcl_loss, cls_post, cam_prob, seg_scores = model(img, label, sp, sal, warmup=True)
        segs.append(seg)
        cls_labels.append(label.max(dim=1)[1].cpu().numpy())

    conv5s = np.concatenate(conv5s,axis=0)
    print(conv5s.shape)
    zmeans = np.concatenate(zmeans,axis=0)
    print(zmeans.shape)
    segs = torch.cat(segs,dim=0)
    labels = F.interpolate(segs.unsqueeze(1), (14,14), mode='nearest').permute(0,2,3,1).flatten().numpy()
    print(labels.shape)
    cls_labels = np.concatenate(cls_labels)

    x_orig = conv5s
    x_reduced= zmeans
    y = labels
    print(np.unique(y))
    y[y==255]=21

    y1 = y * 2
    cls_label = cls_labels.reshape(-1, 1).repeat(14*14,1).flatten() * 2 + 1
    y1[y==0] += cls_label[y==0]

    y2 = y
    cls_label = cls_labels.reshape(-1, 1).repeat(14*14,1).flatten() +22
    y2[y==0] += cls_label[y==0]

    

    return x_orig, x_reduced, y1, y2

def pca_transform(x):
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(x)
    return pca_result_50

def tsne_tranform(x):
    tsne_2 = TSNE(random_state = 42, n_components=2,verbose=0,perplexity=50, n_iter=1000)
    tsne_result_2 = tsne_2.fit_transform(x)
    return tsne_result_2


def visualize(tsne, y, path):
    plt.figure()
    plt.scatter(tsne[:, 0], tsne[:, 1], s=5, c=y, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(43)-0.5).set_ticks(np.arange(42))
    plt.title('Visualizing Kannada MNIST through t-SNE', fontsize=24)
    plt.savefig(path)

if __name__ == "__main__":

    model = gvae_cam(pretrained=True, cam_net_pretrained='pretrained/cam_net.pth', feat_dim=1024, embed_dim=256, n_cluster=4, num_classes=21, 
                     use_bg_prior=False, use_learnable_mix_ratio=False, use_learnable_var=True, wta=False, return_post=False, use_one_hot=True,
                     wcl_loss=False, k=1, bg_thresh=0.5, use_sal=True)
    dataset = VOC12Dataset(img_name_list_path='metadata/voc12/train.txt', voc12_root='/home/dogglas/mil/datasets/VOC2012', input_size=112)
    ckpt = torch.load('gvae/checkpoint-19(1).pth')['model']
    model.load_state_dict(ckpt, strict=False)
    model.train_gen()
    x_orig, x_reduced, y, y2 = prepare_numpy(model, dataset, range(5))

    tsne_orig = tsne_tranform(x_orig)
    visualize(tsne_orig, y, 'tsne_orig')
    visualize(tsne_orig, y2, 'tsne_orig2')

    # tsne_reduced = tsne_tranform(x_reduced)
    # visualize(tsne_reduced, y, 'tsne_reduced')

