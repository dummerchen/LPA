import torch
import torch.nn as nn
from utils import KMeans, fit_kmeans
from torch.nn import functional as F
import numpy as np

def tensor_prompt(a, b, c=None, ortho='uniform'):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a, b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a, b, c), requires_grad=True)
    if ortho == 'ortho':
        nn.init.orthogonal_(p)
    elif ortho == 'zero':
        nn.init.zeros_(p)
    else:
        nn.init.uniform_(p)
    return p


class Promptv6_1_2_3(nn.Module):
    def __init__(self, prompt_layers, use_prompt_block, n_tasks=4, prompt_size=[20], freeze_extract=['msr'],
                 temperature=[1.0], prompt_length=[1], key_dim=[512], emb_dim=[180], enable_lora=[False, True, True], **kwargs):
        super().__init__()
        self.task_id = 0
        self.n_task = n_tasks
        self.key_dim = key_dim
        self.emb_dim = emb_dim
        self.single_task_prompt_pool_size = prompt_size
        self.topK = self.single_task_prompt_pool_size[0]
        self.prompt_length = prompt_length
        self.prompt_layers = prompt_layers
        self.prompt_block = use_prompt_block
        # self.pnum = kwargs.get('pnum') if kwargs.get('pnum', None) is not None else len(prompt_size)
        self.pnum = len(prompt_size)
        self.freeze_extract = freeze_extract
        self.temperature = temperature
        print('temperature', self.temperature)
        for t in range(self.n_task):
            for pid in range(len(self.single_task_prompt_pool_size)):
                for pi in self.prompt_layers[pid]:
                    for pj in self.prompt_block[pid]:
                        PA = tensor_prompt(self.single_task_prompt_pool_size[pid],
                                           self.prompt_length[pid] * sum(enable_lora), self.emb_dim[pid])
                        PB = tensor_prompt(self.prompt_length[pid], self.emb_dim[pid] * sum(enable_lora), ortho='zero')
                        name = str(t) + str(pid) + str(pi) + str(pj)
                        setattr(self, f'pa_{name}', PA)
                        setattr(self, f'pb_{name}', PB)

    def process_task_add(self, task_id, flag='nrdem'):
        def fit_kmeans(features, clusters, length, random_state=3407):
            kmeans = KMeans(n_clusters=clusters, random_state=random_state)
            kmeans.fit(features)
            prompt_kmeans_inits = []
            if length > 1:
                for i in range(clusters):
                    prompt_features = features[kmeans.labels_ == i]
                    if len(prompt_features) > length:
                        pool_kmeans = KMeans(n_clusters=length, random_state=random_state)
                        pool_kmeans.fit(prompt_features)

                        prompt_kmeans = pool_kmeans.cluster_centers_
                        prompt_kmeans_inits.append(prompt_kmeans)
                    else:
                        prompt_kmeans = torch.tile(kmeans.cluster_centers_[i], (length, 1))
                        prompt_kmeans_inits.append(prompt_kmeans)
                out = torch.stack(prompt_kmeans_inits)
            else:
                if clusters == 1:
                    out = kmeans.cluster_centers_
                else:
                    out = kmeans.cluster_centers_.squeeze()
            return out
        self.task_id = task_id
        self.task_features = []
        normalizer = lambda x: x / (torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-10)
        prepos_feat = lambda x: normalizer(x)
        np.random.seed(3407)
        for extract_id in range(len(self.freeze_extract)):
            tmp = []
            print('now load for task {}'.format(self.freeze_extract[extract_id]))
            for i in range(self.task_id+1):
                pre_features = np.load('./{}.npy'.format(flag[extract_id][i]))
                clu = fit_kmeans(features=pre_features, clusters=self.single_task_prompt_pool_size[extract_id], length=1)
                print('task {} now init cluster with {}:{} features'.format(task_id, flag[extract_id][i], len(pre_features)))
                tmp.append(prepos_feat(clu))
                # tmp.append(clu)
            self.task_features.append(tmp)

        for t in range(self.n_task):
            for pid in range(len(self.single_task_prompt_pool_size)):
                for pi in self.prompt_layers[pid]:
                    for pj in self.prompt_block[pid]:
                        PA = getattr(self, f'pa_{str(t) + str(pid) + str(pi) + str(pj)}')
                        PB = getattr(self, f'pb_{str(t) + str(pid) + str(pi) + str(pj)}')
                        if t == self.task_id:
                            PA.requires_grad = True
                            PB.requires_grad = True
                        else:
                            PA.requires_grad = False
                            PB.requires_grad = False
    def forward(self, x_querry, l: str, pid=0):
        x_q = x_querry[pid]
        if pid == 0:
            #msr
            train_features = torch.stack(self.task_features[pid], dim=0).to(x_q.device)
            # B,1,1,d x 1, t, k d => B, 1, 1, k
            cos = torch.cdist(x_q.unsqueeze(1).unsqueeze(2), train_features.unsqueeze(0))
            cos = cos.squeeze(2)
            B, t, k = cos.shape
            cos = cos.view(B, -1)**self.temperature[pid]
            cos = F.softmax(cos, dim=1)

            self.current_task_id = int(cos.argmax() // self.single_task_prompt_pool_size[pid])
            cos = cos.view(B, t, k)
        elif pid == 1:
            # vision encoder
            train_features = torch.stack(self.task_features[pid], dim=0).to(x_q.device)
            cos = torch.einsum('b d, t K d ->b t K', x_q, train_features)
            B, t, k = cos.shape
            cos = cos.view(B, -1)**self.temperature[pid]
            cos = F.softmax(cos, dim=1)
            cos = cos.view(B, t, k)

        pas = []
        pbs = []
        for i in range(len(self.task_features[0])):
            pa = getattr(self, f'pa_{str(i) + str(pid) + l}')
            pb = getattr(self, f'pb_{str(i) + str(pid) + l}')
            pas.append(pa)
            pbs.append(pb)
        PA = torch.stack(pas, dim=0)
        PB = torch.stack(pbs, dim=0)
        pa = torch.einsum('b t k, t k e d -> b t e d', cos, PA)
        pb = PB.transpose(-2, -1).unsqueeze(-1)
        return [pa, pb]
