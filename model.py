import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np
import torch.nn.functional as F
import sys
from parse import parse_args

args = parse_args()


class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError
    
class PureMF(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(PureMF, self).__init__()
        self.num_users  = dataset.n_users
        self.num_items  = dataset.m_items
        self.latent_dim = config['latent_dim_rec']
        self.f = nn.Sigmoid()
        self.__init_weight()
        
    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        print("using Normal distribution N(0,1) initialization for PureMF")
        
    def getUsersRating(self, users):
        users = users.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item.weight
        scores = torch.matmul(users_emb, items_emb.t())
        return self.f(scores)
    
    def bpr_loss(self, users, pos, neg, batch_i):
        users_emb = self.embedding_user(users.long())
        pos_emb   = self.embedding_item(pos.long())
        neg_emb   = self.embedding_item(neg.long())
        pos_scores = torch.sum( (users_emb*pos_emb) / (torch.norm(users_emb)*torch.norm(pos_emb)), dim=1)
        neg_scores = torch.sum( (users_emb*neg_emb) / (torch.norm(users_emb)*torch.norm(neg_emb)), dim=1)
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))

        reg_loss = (1/2)*(users_emb.norm(2).pow(2) + pos_emb.norm(2).pow(2) + neg_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss
        
    def forward(self, users, items):
        users = users.long()
        items = items.long()
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        scores = torch.sum(users_emb*items_emb, dim=1)
        return self.f(scores)

class LightGCN(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()

        if args.model in ['lgn', 'lgn-navip']:
            self.Graph = self.dataset.getSparseGraph_lgn()
        elif args.model in ['lgn-apda']:
            self.Graph = self.dataset.getSparseGraph_navip()
        elif args.model == 'lgn-adjnorm':
            self.Graph = self.dataset.getSparseGraph_adjnorm()
        elif args.model == 'lgn-pc':
            self.Graph, self.rowsum = self.dataset.getSparseGraph_pc()
        elif args.model == 'lgn-reg':
            self.Graph, self.rowsum = self.dataset.getSparseGraph_pc()
        elif args.model == 'lgn-macr':
            self.Graph, self.rowsum = self.dataset.getSparseGraph_pc()
        elif args.model == 'ours':
            self.Graph = self.dataset.getSparseGraph_lgn()
            self.embed_user_first = torch.Tensor(np.load('lgn_embed_user_'+args.dataset+'.npy'))
            self.embed_item_first = torch.Tensor(np.load('lgn_embed_item_'+args.dataset+'.npy'))

            self.sim_score = self.f(torch.mm(self.embed_user_first, self.embed_item_first.T))
            self.alpha = args.alpha
            print("alpha: ", self.alpha)
            self.exp_prob = torch.max(self.alpha * torch.ones([self.num_users, self.num_items]), self.sim_score)
            print("self.exp_prob, min, max: ", torch.min(self.exp_prob), torch.max(self.exp_prob))
            print("Exposure probability matrix is computed")

        print(f"lgn is already to go(dropout:{self.config['dropout']})")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config['dropout']:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        # APDA
        if args.model == 'lgn-apda':
            all_emb_new = all_emb
            cof_lambda = 0.6
            for layer in range(self.n_layers):
                if self.A_split:
                    temp_emb = []
                    for f in range(len(g_droped)):
                        temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                    side_emb = torch.cat(temp_emb, dim=0)
                    all_emb = side_emb
                else:
                    all_emb_new = all_emb_new + cof_lambda * all_emb
                    all_emb_new = torch.sparse.mm(g_droped, all_emb_new)
                all_emb_new_norm = F.normalize(all_emb_new, p=2, dim=1)
                embs.append(all_emb_new_norm)
        else:
            for layer in range(self.n_layers):
                if self.A_split:
                    temp_emb = []
                    for f in range(len(g_droped)):
                        temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                    side_emb = torch.cat(temp_emb, dim=0)
                    all_emb = side_emb
                else:
                    all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg, batch_i):
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                         posEmb0.norm(2).pow(2)  +
                         negEmb0.norm(2).pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        # Debiased LightGCN
        if args.model == 'ours':
            propensity_scores = 1.0 / self.exp_prob[users.long(), pos.long()].to(pos_scores.device)
            loss = torch.mean(propensity_scores * torch.nn.functional.softplus(neg_scores - pos_scores))
        elif args.model == 'lgn-pc':
            pc_alpha = 1.0
            degree_mat = torch.from_numpy(self.rowsum).to(pos_scores.device)
            threshold = torch.ones(pos.shape).to(pos_scores.device)
            threshold = threshold * 1e-5
            aaa = torch.squeeze(degree_mat[pos])
            bbb = torch.squeeze(degree_mat[neg])
            pos_scores = pos_scores + pc_alpha * 1.0 / torch.max(aaa, threshold)
            neg_scores = neg_scores + pc_alpha * 1.0 / torch.max(bbb, threshold)
            loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        elif args.model == 'lgn-reg':
            rec_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
            cof_gamma = 1e-4
            degree_mat = torch.from_numpy(self.rowsum).to(pos_scores.device)
            aaa = torch.squeeze(degree_mat[pos])
            pcc_loss = torch.cosine_similarity(pos_scores, aaa, dim=0)
            loss = rec_loss + cof_gamma * pcc_loss
        elif args.model == 'lgn-macr':
            macr_alpha = 1.0
            macr_beta = 1.0
            eps = 1e-7
            degree_mat = torch.from_numpy(self.rowsum).to(pos_scores.device)
            degree_mat = self.f(degree_mat)
            pos_scores = self.f(pos_scores)
            neg_scores = self.f(neg_scores)
            rec_loss = torch.mean( - torch.log(pos_scores + eps) - torch.log(1-neg_scores + eps) )
            item_loss = torch.mean( - torch.log(degree_mat[pos] + eps) - torch.log(1-degree_mat[neg] + eps) )
            user_loss = torch.mean( - torch.log(degree_mat[users] + eps) - torch.log(1-degree_mat[users] + eps) )
            loss = rec_loss + macr_alpha*item_loss + macr_beta*user_loss
        # LightGCN
        else:
            loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma