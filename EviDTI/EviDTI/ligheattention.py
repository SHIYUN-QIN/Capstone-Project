import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from compound_encode import AtomEmbedding, BondEmbedding, BondFloatRBF
from torch_geometric.nn import TransformerConv, global_max_pool as gmp,global_mean_pool
import os
# from torch_scatter import scatter

device = 'cuda:1'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
class LightAttention(nn.Module):
    def __init__(self, embeddings_dim=1024, output_dim=2, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.atom_names = ["atomic_num", "formal_charge", "degree",
                       "chiral_tag", "total_numHs", "is_aromatic",
                       "hybridization"]
        self.bond_names = ["bond_dir", "bond_type", "is_in_ring"]
        self.bond_float_names = ['bond_length']
        self.bond_angle_float_names = ['bond_angle']

        self.embed_dim = 32
        self.dropout_rate = 0.2
        self.layer_num = 8
        self.readout = "mean"
        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)
        self.d_conv1 = nn.Sequential(nn.Conv1d(in_channels=1,out_channels=8,
                                               kernel_size=3,stride=1,padding=1),
                                     nn.ReLU(),
                                     nn.MaxPool1d(kernel_size=2))
        self.d_conv2 = nn.Sequential(nn.Conv1d(8,16,3,1,1),
                                     nn.ReLU(),
                                     nn.MaxPool1d(2))
        self.d_fc = nn.Sequential(nn.Linear(256, 64),
                                   nn.Dropout(dropout),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(64))
        self.init_atom_embedding = AtomEmbedding(self.atom_names, self.embed_dim)
        self.init_bond_embedding = BondEmbedding(self.bond_names, self.embed_dim)
        self.init_bond_float_rbf = BondFloatRBF(self.bond_float_names, self.embed_dim)
        # self.graph_pool = gmp()
        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 64),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )

        self.abg_conv1 = TransformerConv(in_channels=7, out_channels=32, heads=1,
                                         dropout=0.2, edge_dim=4)
        self.abg_conv2 = TransformerConv(in_channels=32, out_channels=64, heads=1,
                                         dropout=0.2, edge_dim=32)
        self.abg_conv3 = TransformerConv(in_channels=64, out_channels=128, heads=1,
                                         dropout=0.2, edge_dim=64)
        self.bag_conv1 = TransformerConv(in_channels=4, out_channels=32, heads=1,
                                         dropout=0.2, edge_dim=1)
        self.bag_conv2 = TransformerConv(in_channels=32, out_channels=64, heads=1,
                                         dropout=0.2, edge_dim=1)
        self.bag_conv3 = TransformerConv(in_channels=64, out_channels=128, heads=1,
                                         dropout=0.2, edge_dim=1)
        self.tg_conv1 = TransformerConv(in_channels=1, out_channels=32, heads=1,
                                         dropout=0.2, edge_dim=1)
        self.tg_conv2 = TransformerConv(in_channels=32, out_channels=64, heads=1,
                                         dropout=0.2, edge_dim=1)
        self.tg_conv3 = TransformerConv(in_channels=64, out_channels=128, heads=1,
                                         dropout=0.2, edge_dim=1)

        self.abg_fc1 = torch.nn.Linear(32 * 4, 1024)
        self.abg_bn1 = nn.BatchNorm1d(1024)
        self.abg_fc2 = torch.nn.Linear(1024, 64)
        self.abg_bn2 = nn.BatchNorm1d(64)

        self.tg_fc1 = torch.nn.Linear(32 * 4, 1024)
        self.tg_bn1 = nn.BatchNorm1d(1024)
        self.tg_fc2 = torch.nn.Linear(1024, 64)
        self.tg_bn2 = nn.BatchNorm1d(64)

        self.output = nn.Linear(64*4, output_dim)
        self.final_activation = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, batch, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequecnes in the batch. All values corresponding to padding are False and the rest is True.

        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        data = batch
        x, edge_index, batch, edge_attr_list = data.x, data.edge_index,data.batch, data.edge_attr
        batch = batch.to(device)
        t_1D_embedding = data.t_1D_feature
        d_2D_embedding = data.d_2D_feature
        bag = data.bag
        t_graph = data.t_3D_graph
        bag_x = torch.Tensor([])
        bag_edge_index = torch.Tensor([])
        bag_edge_attr = torch.Tensor([])
        for g in bag:
            g_x = torch.tensor(g.x)
            bag_x = torch.cat((bag_x,g_x))
            g_edge_index = torch.tensor(g.edge_index)
            bag_edge_index = torch.cat((bag_edge_index,g_edge_index),dim=1)
            g_edge_attr = torch.tensor(g.edge_attr)
            bag_edge_attr = torch.cat((bag_edge_attr,g_edge_attr))
        bag_x = bag_x.to(device).to(torch.float32)
        bag_edge_index = bag_edge_index.to(device).to(torch.int64)
        bag_edge_attr = bag_edge_attr.to(device).to(torch.float32)
        bag_edge_attr = torch.unsqueeze(bag_edge_attr, dim=1)

        tg_x = torch.Tensor([])
        tg_edge_index = torch.Tensor([])
        tg_edge_attr = torch.Tensor([])
        tg_batch = torch.Tensor([])
        i = 0
        for t in t_graph:
            t_x = torch.tensor(t.x)
            tg_x = torch.cat((tg_x,t_x))
            temp = np.full(len(t_x), i)
            tg_batch = torch.cat((tg_batch,torch.tensor(temp)))
            tg_edge_index = torch.tensor(t.edge_index)
            tg_edge_index = torch.cat((tg_edge_index,tg_edge_index),dim=1)
            tg_edge_attr = torch.tensor(t.edge_attr)
            tg_edge_attr = torch.cat((tg_edge_attr,tg_edge_attr))
            i+=1
        tg_x = tg_x.to(device).to(torch.float32)
        tg_edge_index = tg_edge_index.to(device).to(torch.int64)
        tg_edge_attr = tg_edge_attr.to(device).to(torch.float32)
        tg_edge_attr = torch.unsqueeze(tg_edge_attr, dim=1)
        tg_batch = tg_batch.to(device).to(torch.int64)


        x = x.to(device).to(torch.float32)
        edge_index = edge_index.to(device)

        # d_node_hidden = self.init_atom_embedding(d_3D_embedding.)
        t_1D = [torch.tensor(item) for item in t_1D_embedding]
        t_1D= pad_sequence(t_1D, batch_first=True)
        t_1D = t_1D.permute(0, 2, 1)
        t_1D = t_1D.to(device).to(torch.float32)
        d_2D= torch.tensor(np.array([item for item in d_2D_embedding]))
        d_2D = d_2D.to(device).to(torch.float32)
        a = []
        a = torch.Tensor(a)
        for item in edge_attr_list:
            temp = torch.tensor(item)
            a = torch.cat((a, temp))

        edge_attr_list = a.to(device).to(torch.float32)

        # target lightattention
        t_o = self.feature_convolution(t_1D)  # [batch_size, embeddings_dim, sequence_length]
        t_o = self.dropout(t_o)  # [batch_gsize, embeddings_dim, sequence_length]
        attention = self.attention_convolution(t_1D)  # [batch_size, embeddings_dim, sequence_length]

        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        attention = attention.masked_fill(mask[:, None, :] == False, -1e9)

        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)

        t_o1 = torch.sum(t_o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
        t_o2, _ = torch.max(t_o, dim=-1)  # [batchsize, embeddings_dim]
        t_o = torch.cat([t_o1, t_o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        t_o = self.linear(t_o)  # [batchsize, 64]

        # drug_2d
        d_o = self.d_fc(d_2D)
        # drug_3D
        # node_hidden = self.init_atom_embedding(x)
        atom_h = self.abg_conv1(x,edge_index,edge_attr_list)
        atom_h = self.relu(atom_h)
        edge_h = self.bag_conv1(bag_x,bag_edge_index,bag_edge_attr)
        edge_h = self.relu(edge_h)
        atom_h = self.abg_conv2(atom_h,edge_index,edge_h)
        atom_h = self.relu(atom_h)
        edge_h = self.bag_conv2(edge_h, bag_edge_index, bag_edge_attr)
        edge_h = self.relu(edge_h)
        atom_h = self.abg_conv3(atom_h,edge_index,edge_h)
        atom_h = self.relu(atom_h)
        edge_h = self.bag_conv3(edge_h, bag_edge_index, bag_edge_attr)
        edge_h = self.relu(edge_h)
        atom_h = gmp(atom_h,batch)
        atom_h = self.abg_fc1(atom_h)
        atom_h = self.abg_bn1(atom_h)
        atom_h = self.abg_fc2(atom_h)
        atom_h = self.abg_bn2(atom_h)

        AA_h = self.tg_conv1(tg_x, tg_edge_index, tg_edge_attr)
        AA_h = self.relu(AA_h)
        AA_h = self.tg_conv2(AA_h, tg_edge_index, tg_edge_attr)
        AA_h = self.relu(AA_h)
        AA_h = self.tg_conv3(AA_h, tg_edge_index, tg_edge_attr)
        AA_h = self.relu(AA_h)
        # size = int(batch.max().item() + 1)
        # AA_h = scatter(AA_h, batch, dim=0, dim_size=size, reduce='max')
        AA_h = global_mean_pool(AA_h, tg_batch)
        AA_h = self.tg_fc1(AA_h)
        AA_h = self.tg_bn1(AA_h)
        AA_h = self.tg_fc2(AA_h)
        AA_h = self.tg_bn2(AA_h)

        # cat
        cat_v = torch.cat((t_o, d_o, atom_h, AA_h), 1)
        cat_v = self.output(cat_v) # [batchsize, output_dim]
        output = nn.functional.softplus(cat_v) + 1
        return output
