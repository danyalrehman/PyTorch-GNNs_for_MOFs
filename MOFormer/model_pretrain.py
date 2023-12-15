import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, feature_dimension, num_features_len):
        super(ConvLayer, self).__init__()
        self.feature_dimension = feature_dimension
        self.num_features_len = num_features_len
        self.fully_connected_layers = nn.Linear(2*self.feature_dimension+self.num_features_len, 2*self.feature_dimension)
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.batch_normalization_first = nn.BatchNorm1d(2*self.feature_dimension)
        self.batch_normalization_second = nn.BatchNorm1d(self.feature_dimension)

    def forward(self, input_features, num_features, feature_index):
        N, M = feature_index.shape
        # convolution
        atom_num_features = input_features[feature_index, :]
        total_num_features = torch.cat([input_features.unsqueeze(1).expand(N, M, self.feature_dimension), atom_num_features, num_features], dim=2)
        total_gated_fea = self.fully_connected_layers(total_num_features)
        total_gated_fea = self.batch_normalization_first(total_gated_fea.view( -1, self.feature_dimension*2)).view(N, M, self.feature_dimension*2)
        
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        
        nbr_core = self.softplus(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.batch_normalization_second(nbr_sumed)
        out = self.softplus(input_features + nbr_sumed)
        
        return out


class CrystalGraphConvNet(nn.Module):
    def __init__(self, orig_feature_dimension, num_features_len,
                 feature_dimension=64, n_conv=3, h_fea_len=128, n_h=1, 
                 dropout=0):

        super(CrystalGraphConvNet, self).__init__()
        self.dropout = dropout
        self.embedding = nn.Linear(orig_feature_dimension, feature_dimension)
        self.convs = nn.ModuleList([ConvLayer(feature_dimension=feature_dimension, num_features_len=num_features_len) for _ in range(n_conv)])
        
        self.conv_to_fc = nn.Linear(feature_dimension, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()

        self.fc_head = nn.Sequential(
            nn.Linear(h_fea_len, h_fea_len), 
            nn.Softplus(),
            nn.Linear(h_fea_len, h_fea_len)
        )



    def forward(self, atomic_features, num_features, feature_index, crystal_index):
        atomic_features = self.embedding(atomic_features)
        for conv_func in self.convs:
            atomic_features = conv_func(atomic_features, num_features, feature_index)
        crystal_features = self.pooling(atomic_features, crystal_index)
        crystal_features = self.conv_to_fc(self.conv_to_fc_softplus(crystal_features))
        crystal_features = self.conv_to_fc_softplus(crystal_features)

        return self.fc_head(crystal_features)


    def pooling(self, atomic_features, crystal_index):
        summed_fea = [torch.mean(atomic_features[idx_map], dim=0, keepdim=True) for idx_map in crystal_index]
        return torch.cat(summed_fea, dim=0)