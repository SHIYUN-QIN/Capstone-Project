from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric import data as DATA
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from utils import proprocess,KIBA_proprocess,davis_proprocess,case_proprocess

class bond_angle_graph_data(InMemoryDataset):

    def __init__(self, root, d_2D_path, t_1D_path, d_3D_path, label_file, metadata_path,
                 transform=None, pre_transform=None, pre_filter=None):
        self.d_2D_path = d_2D_path
        self.t_1D_path = t_1D_path
        self.d_3D_path = d_3D_path
        self.label_file = label_file
        self.metadata_path = metadata_path
        super(bond_angle_graph_data, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['drugbank.pt']
        # return ['gcn.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        # d_2D_path = 'dataset/bond_angle/d_feature.npy'
        # t_1D_path = 'dataset/bond_angle/t_feature.npy'
        # d_3D_path = 'dataset/bond_angle/d_3D_feature.npy'
        # t_3D_path = 'dataset/bond_angle/t_3d_feature.npy'
        d_2D_path = self.d_2D_path
        t_1D_path = self.t_1D_path
        d_3D_path = self.d_3D_path
        label_file = self.label_file
        metadata_path = self.metadata_path
        atom_bond_graph_list = []
        d_2D_feature, t_1D_feature, d_3D_feature, label, metadata_list = proprocess(d_2D_path, t_1D_path,
                                                                          d_3D_path,label_file,metadata_path)
        data_len = len(label)
        # d_2D_fea = []
        for i in range(data_len):
            graph_dict = {}

            d_2D_fea = d_2D_feature[i]

            # drug 2D GCN
            # for d_2D_smile in d_2D_feature:
            # mol = Chem.MolFromSmiles(d_2D_feature[i])
            # if mol is not None:
            #     data = mol_to_graph_data_obj(mol)
            #     d_2D_fea = data

            t_1D_fea = t_1D_feature[i]
            atom_feature = np.stack([d_3D_feature[i]['atomic_num'], d_3D_feature[i]['chiral_tag'],
                                     d_3D_feature[i]['degree'], d_3D_feature[i]['explicit_valence'],
                                     d_3D_feature[i]['formal_charge'], d_3D_feature[i]['hybridization'],
                                     d_3D_feature[i]['implicit_valence']])
            bond_feature = np.stack([d_3D_feature[i]['bond_dir'], d_3D_feature[i]['bond_type'],
                                     d_3D_feature[i]['is_in_ring'], d_3D_feature[i]['bond_length']])
            angle_feature = d_3D_feature[i]['bond_angle']

            atom_bond_graph = DATA.Data(x=torch.tensor(atom_feature).transpose(1, 0),
                                        edge_index=torch.LongTensor(d_3D_feature[i]['edges'].transpose(1, 0)),
                                        edge_attr=bond_feature.T)
            atom_bond_graph.bag = DATA.Data(x=torch.tensor(bond_feature).transpose(1, 0),
                                            edge_index=torch.LongTensor(
                                                d_3D_feature[i]['BondAngleGraph_edges'].transpose(1, 0)),
                                            edge_attr=angle_feature)

            atom_bond_graph.d_2D_feature = d_2D_fea
            atom_bond_graph.t_1D_feature = t_1D_fea
            # AA_feature = t_3D_feature[i]['fp']
            # AA_feature = torch.tensor(AA_feature)
            # AA_feature = torch.unsqueeze(AA_feature, dim=1)
            # AA_adj = t_3D_feature[i]['adj']
            # AA_edge_attr = torch.tensor(t_3D_feature[i]['edge_attr'])
            # AA_edge_attr = torch.unsqueeze(AA_edge_attr, dim=1)

            # atom_bond_graph.t_3D_graph = DATA.Data(x=AA_feature, edge_index=torch.LongTensor(
            #                                     AA_adj).transpose(1, 0),
            #                        edge_attr=AA_edge_attr)

            atom_bond_graph.l = label[i]
            atom_bond_graph.metadata = metadata_list[i]
            # graph_dict['d_3D_graph'] = atom_bond_graph
            # graph_dict['t_3D_graph'] = t_3D_graph

            atom_bond_graph_list.append(atom_bond_graph)

        if self.pre_filter is not None:
            atom_bond_graph_list = [data for data in atom_bond_graph_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            atom_bond_graph_list = [self.pre_transform(data) for data in atom_bond_graph_list]

        data_1, slices_1 = self.collate(atom_bond_graph_list)
        torch.save((data_1, slices_1), self.processed_paths[0])

class KIBA_graph_data(InMemoryDataset):

    def __init__(self, root, d_2D_path, t_1D_path, d_3D_path, label_file, metadata_path,
                 transform=None, pre_transform=None, pre_filter=None):
        self.d_2D_path = d_2D_path
        self.t_1D_path = t_1D_path
        self.d_3D_path = d_3D_path
        self.label_file = label_file
        self.metadata_path = metadata_path
        super(KIBA_graph_data, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['KIBA.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        d_2D_path = self.d_2D_path
        t_1D_path = self.t_1D_path
        d_3D_path = self.d_3D_path
        label_file = self.label_file
        metadata_path = self.metadata_path
        atom_bond_graph_list = []
        d_2D_feature, t_1D_feature, d_3D_feature, label, metadata_list = KIBA_proprocess(d_2D_path, t_1D_path,
                                                                          d_3D_path,label_file,metadata_path)
        print(len(d_2D_feature))
        print(len(label))
        data_len = len(label)
        for i in range(data_len):
            graph_dict = {}
            d_2D_fea = d_2D_feature[i]
            t_1D_fea = t_1D_feature[i]
            atom_feature = np.stack([d_3D_feature[i]['atomic_num'], d_3D_feature[i]['chiral_tag'],
                                     d_3D_feature[i]['degree'], d_3D_feature[i]['explicit_valence'],
                                     d_3D_feature[i]['formal_charge'], d_3D_feature[i]['hybridization'],
                                     d_3D_feature[i]['implicit_valence']])
            bond_feature = np.stack([d_3D_feature[i]['bond_dir'], d_3D_feature[i]['bond_type'],
                                     d_3D_feature[i]['is_in_ring'], d_3D_feature[i]['bond_length']])
            angle_feature = d_3D_feature[i]['bond_angle']

            atom_bond_graph = DATA.Data(x=torch.tensor(atom_feature).transpose(1, 0),
                                        edge_index=torch.LongTensor(d_3D_feature[i]['edges'].transpose(1, 0)),
                                        edge_attr=bond_feature.T)
            atom_bond_graph.bag = DATA.Data(x=torch.tensor(bond_feature).transpose(1, 0),
                                            edge_index=torch.LongTensor(
                                                d_3D_feature[i]['BondAngleGraph_edges'].transpose(1, 0)),
                                            edge_attr=angle_feature)

            atom_bond_graph.d_2D_feature = d_2D_fea
            atom_bond_graph.t_1D_feature = t_1D_fea
            # AA_feature = t_3D_feature[i]['fp']
            # AA_feature = torch.tensor(AA_feature)
            # AA_feature = torch.unsqueeze(AA_feature, dim=1)
            # AA_adj = t_3D_feature[i]['adj']
            # AA_edge_attr = torch.tensor(t_3D_feature[i]['edge_attr'])
            # AA_edge_attr = torch.unsqueeze(AA_edge_attr, dim=1)

            # atom_bond_graph.t_3D_graph = DATA.Data(x=AA_feature, edge_index=torch.LongTensor(
            #                                     AA_adj).transpose(1, 0),
            #                        edge_attr=AA_edge_attr)

            atom_bond_graph.l = label[i]
            atom_bond_graph.metadata = metadata_list[i]
            # graph_dict['d_3D_graph'] = atom_bond_graph
            # graph_dict['t_3D_graph'] = t_3D_graph

            atom_bond_graph_list.append(atom_bond_graph)

        if self.pre_filter is not None:
            atom_bond_graph_list = [data for data in atom_bond_graph_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            atom_bond_graph_list = [self.pre_transform(data) for data in atom_bond_graph_list]

        data_1, slices_1 = self.collate(atom_bond_graph_list)
        torch.save((data_1, slices_1), self.processed_paths[0])


class davis_graph_data(InMemoryDataset):

    def __init__(self, root, d_2D_path, t_1D_path, d_3D_path, label_file, metadata_path,
                 transform=None, pre_transform=None, pre_filter=None):
        self.d_2D_path = d_2D_path
        self.t_1D_path = t_1D_path
        self.d_3D_path = d_3D_path
        self.label_file = label_file
        self.metadata_path = metadata_path
        super(davis_graph_data, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['davis.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        d_2D_path = self.d_2D_path
        t_1D_path = self.t_1D_path
        d_3D_path = self.d_3D_path
        label_file = self.label_file
        metadata_path = self.metadata_path
        atom_bond_graph_list = []
        d_2D_feature, t_1D_feature, d_3D_feature, label, metadata_list = davis_proprocess(d_2D_path, t_1D_path,
                                                                          d_3D_path,label_file,metadata_path)
        print(len(d_2D_feature))
        print(len(label))
        data_len = len(label)
        for i in range(data_len):
            graph_dict = {}
            d_2D_fea = d_2D_feature[i]
            t_1D_fea = t_1D_feature[i]
            atom_feature = np.stack([d_3D_feature[i]['atomic_num'], d_3D_feature[i]['chiral_tag'],
                                     d_3D_feature[i]['degree'], d_3D_feature[i]['explicit_valence'],
                                     d_3D_feature[i]['formal_charge'], d_3D_feature[i]['hybridization'],
                                     d_3D_feature[i]['implicit_valence']])
            bond_feature = np.stack([d_3D_feature[i]['bond_dir'], d_3D_feature[i]['bond_type'],
                                     d_3D_feature[i]['is_in_ring'], d_3D_feature[i]['bond_length']])
            angle_feature = d_3D_feature[i]['bond_angle']

            atom_bond_graph = DATA.Data(x=torch.tensor(atom_feature).transpose(1, 0),
                                        edge_index=torch.LongTensor(d_3D_feature[i]['edges'].transpose(1, 0)),
                                        edge_attr=bond_feature.T)
            atom_bond_graph.bag = DATA.Data(x=torch.tensor(bond_feature).transpose(1, 0),
                                            edge_index=torch.LongTensor(
                                                d_3D_feature[i]['BondAngleGraph_edges'].transpose(1, 0)),
                                            edge_attr=angle_feature)

            atom_bond_graph.d_2D_feature = d_2D_fea
            atom_bond_graph.t_1D_feature = t_1D_fea

            atom_bond_graph.l = label[i]
            atom_bond_graph.metadata = metadata_list[i]

            atom_bond_graph_list.append(atom_bond_graph)

        if self.pre_filter is not None:
            atom_bond_graph_list = [data for data in atom_bond_graph_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            atom_bond_graph_list = [self.pre_transform(data) for data in atom_bond_graph_list]

        data_1, slices_1 = self.collate(atom_bond_graph_list)
        torch.save((data_1, slices_1), self.processed_paths[0])


class case_graph_data(InMemoryDataset):

    def __init__(self, root, d_2D_path, t_1D_path, d_3D_path, label_file, metadata_path,
                 transform=None, pre_transform=None, pre_filter=None):
        self.d_2D_path = d_2D_path
        self.t_1D_path = t_1D_path
        self.d_3D_path = d_3D_path
        self.label_file = label_file
        self.metadata_path = metadata_path
        super(case_graph_data, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['drugbank_case.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        d_2D_path = self.d_2D_path
        t_1D_path = self.t_1D_path
        d_3D_path = self.d_3D_path
        label_file = self.label_file
        metadata_path = self.metadata_path
        atom_bond_graph_list = []
        d_2D_feature, t_1D_feature, d_3D_feature, label, metadata_list = case_proprocess(d_2D_path, t_1D_path,
                                                                          d_3D_path,label_file,metadata_path)
        print(len(d_2D_feature))
        print(len(label))
        data_len = len(label)
        for i in range(data_len):
            graph_dict = {}
            d_2D_fea = d_2D_feature[i]
            t_1D_fea = t_1D_feature[i]
            atom_feature = np.stack([d_3D_feature[i]['atomic_num'], d_3D_feature[i]['chiral_tag'],
                                     d_3D_feature[i]['degree'], d_3D_feature[i]['explicit_valence'],
                                     d_3D_feature[i]['formal_charge'], d_3D_feature[i]['hybridization'],
                                     d_3D_feature[i]['implicit_valence']])
            bond_feature = np.stack([d_3D_feature[i]['bond_dir'], d_3D_feature[i]['bond_type'],
                                     d_3D_feature[i]['is_in_ring'], d_3D_feature[i]['bond_length']])
            angle_feature = d_3D_feature[i]['bond_angle']

            atom_bond_graph = DATA.Data(x=torch.tensor(atom_feature).transpose(1, 0),
                                        edge_index=torch.LongTensor(d_3D_feature[i]['edges'].transpose(1, 0)),
                                        edge_attr=bond_feature.T)
            atom_bond_graph.bag = DATA.Data(x=torch.tensor(bond_feature).transpose(1, 0),
                                            edge_index=torch.LongTensor(
                                                d_3D_feature[i]['BondAngleGraph_edges'].transpose(1, 0)),
                                            edge_attr=angle_feature)

            atom_bond_graph.d_2D_feature = d_2D_fea
            atom_bond_graph.t_1D_feature = t_1D_fea

            atom_bond_graph.l = label[i]
            atom_bond_graph.metadata = metadata_list[i]

            atom_bond_graph_list.append(atom_bond_graph)

        if self.pre_filter is not None:
            atom_bond_graph_list = [data for data in atom_bond_graph_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            atom_bond_graph_list = [self.pre_transform(data) for data in atom_bond_graph_list]

        data_1, slices_1 = self.collate(atom_bond_graph_list)
        torch.save((data_1, slices_1), self.processed_paths[0])