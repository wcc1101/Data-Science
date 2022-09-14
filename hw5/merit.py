import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, inp_size, outp_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, outp_size)
        )

    def forward(self, x):
        return self.net(x)


class GraphEncoder(nn.Module):

    def __init__(self, 
                  gnn,
                  projection_hidden_size,
                  projection_size):
        
        super().__init__()
        
        self.gnn = gnn
        self.projector = MLP(512, projection_size, projection_hidden_size)           
        
    def forward(self, adj, in_feats, sparse):
        representations = self.gnn(in_feats, adj, sparse)
        representations = representations.view(-1, representations.size(-1))
        projections = self.projector(representations)  # (batch, proj_dim)
        return projections

    
class EMA():
    
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def Sim(h1, h2):
    CosSim = nn.CosineSimilarity(dim = 1)
    # 使用pytorch.nn裡的function
    return CosSim(h1, h2)

def phi(h1, h2):
    CosSim1 = nn.CosineSimilarity(dim = 1)
    phi = CosSim1(h1, h1)
    phi *= 0
    CosSim0 = nn.CosineSimilarity(dim = 0)
    for i in range(h1.shape[0]):
        for j in range(h2.shape[0]):
            if j != i:
                phi[i] += torch.exp(CosSim0(h1[i], h2[j]))
    # 照著paper裡的做法
    return phi

def Lcn(h1, h2, z1, z2):
    lcn1 = -torch.log(torch.exp(Sim(h1, z2)) / torch.exp(Sim(h1, z2)).sum(dim = -1))
    lcn2 = -torch.log(torch.exp(Sim(h2, z1)) / torch.exp(Sim(h2, z1)).sum(dim = -1))
    lcn = (lcn1.sum(dim = -1) + lcn2.sum(dim = -1)) / 2
    # 照著paper裡的做法
    return lcn

def Lcv(h1, h2):
    Linter1 = -torch.log(torch.exp(Sim(h1, h2)) / torch.exp(Sim(h1, h2)).sum(dim = -1))
    Lintra1 = -torch.log(torch.exp(Sim(h1, h2)) / (torch.exp(Sim(h1, h2)) + phi(h1, h1)))
    Linter2 = -torch.log(torch.exp(Sim(h1, h2)) / torch.exp(Sim(h1, h2)).sum(dim = -1))
    Lintra2 = -torch.log(torch.exp(Sim(h1, h2)) / (torch.exp(Sim(h1, h2)) + phi(h2, h2)))
    lcv1 = Linter1 + Lintra1
    lcv2 = Linter2 + Lintra2
    lcv = (lcv1.sum(dim = -1) + lcv2.sum(dim = -1)) / 2
    # 照著paper裡的做法
    return lcv

class MERIT(nn.Module):
    
    def __init__(self, 
                 gnn,
                 feat_size,
                 projection_size, 
                 projection_hidden_size,
                 prediction_size,
                 prediction_hidden_size,
                 moving_average_decay,
                 beta):
        
        super().__init__()

        self.online_encoder = GraphEncoder(gnn, projection_hidden_size, projection_size)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(self.target_encoder, False)
        self.target_ema_updater = EMA(moving_average_decay)
        self.online_predictor = MLP(projection_size, prediction_size, prediction_hidden_size)
        self.beta = beta
                   
    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_ma(self):
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, aug_adj_1, aug_adj_2, aug_feat_1, aug_feat_2, sparse):
        ## TODO: 
        # Given training instances: aug_adj_1, aug_adj_2, aug_feat_1, aug_feat_2, hyperparameters: beta
        # Please implement the main algorithm of merit using online_encoder and target_encoder 
        # hint: use self.online_encoder, self.online_predictor, self.target_encoder
        # hint2: remember to detach target network

        # The output should be "the calculated overall contrastive loss of MERIT"
        # Therefore, you should implement your "contrative loss function" first
        # For the CL term, please refer to the released hw powerpoint or the original paper for details

        online_proj_1 = self.online_encoder(aug_adj_1, aug_feat_1, sparse)
        online_proj_2 = self.online_encoder(aug_adj_2, aug_feat_2, sparse)
        with torch.no_grad():
            target_proj_1 = self.target_encoder(aug_adj_1, aug_feat_1, sparse)
            target_proj_2 = self.target_encoder(aug_adj_2, aug_feat_2, sparse)
        online_pred_1 = self.online_predictor(online_proj_1)
        online_pred_2 = self.online_predictor(online_proj_2) #用encoder, predictor算出h1, h2, z1, z2

        L = self.beta * Lcv(online_pred_1, online_pred_2) + (1.0 - self.beta) * Lcn(online_pred_1, online_pred_2, target_proj_1.detach(), target_proj_2.detach())
        # 照著paper裡的做法
        return L.mean()