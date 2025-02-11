import os
import numpy as np
import torch
from torch.utils.data import Dataset
from config import nn_config
data_class = nn_config['pdb_class']
class readData(Dataset): # 用于训练
    def __init__(self, name_list, proj_dir, lig_dict, true_file):
        self.label_dict = self._read(true_file,skew=1)
        if name_list is not None:
            self.name_list = name_list
        self.proj_dir     = proj_dir
        self.lig_dict = lig_dict
    def __len__(self):
        return len(self.name_list)
    def __getitem__(self, idx):
        name, lig = self.name_list[idx] # name是pdb_name
        dssp = self.Normalize(np.load(f'{self.proj_dir}/{data_class}_dssp/{name}.npy'),nn_config[f'dssp_max_repr'],nn_config[f'dssp_min_repr'])
        ankh = self.Normalize(np.load(f'{self.proj_dir}/ankh/{name}.npy'),nn_config[f'ankh_max_repr'],nn_config[f'ankh_min_repr'])
        
        feature  = np.concatenate([dssp,ankh],axis=1)
        ligand = self.Normalize(self.lig_dict[lig], nn_config[f'ion_max_repr'], nn_config[f'ion_min_repr'])# 768
        xyz      = np.load(f'{self.proj_dir}/{data_class}_pos/{name}.npy')
        y_true   = np.asarray(list(self.label_dict[(name,lig)]),dtype=int)
        return feature, ligand, xyz, y_true
    
    def Normalize(self, arr, max_value, min_value):
        scalar = max_value - min_value
        scalar[scalar==0] = 1
        return (arr - min_value) / scalar
    
    def collate_fn(self, batch):
        features, ligands, xyzs, y_trues = zip(*batch)
        maxlen = max(1500,max([f.shape[0] for f in features]))
        batch_feat = []
        batch_ligand = []
        batch_xyz = []
        batch_mask  = []
        batch_y_true = []
        for idx in range(len(batch)):
            batch_feat.append(self._padding(features[idx], maxlen))
            batch_ligand.append(torch.tensor(ligands[idx],dtype=torch.float))
            batch_xyz.append(self._padding(xyzs[idx], maxlen))
            
            mask = np.zeros(maxlen)
            mask[:features[idx].shape[0]] = 1
            batch_mask.append(torch.tensor(mask, dtype = torch.long))
            
            pad_y = np.zeros(maxlen)
            pad_y[:y_trues[idx].shape[0]] = y_trues[idx]
            batch_y_true.append(torch.tensor(pad_y,dtype=torch.float))
            
        return torch.stack(batch_feat), torch.stack(batch_ligand), torch.stack(batch_xyz), torch.stack(batch_mask), torch.stack(batch_y_true)
                  
    def _padding(self, arr, maxlen=1500):
        padded = np.zeros((maxlen,*arr.shape[1:]), dtype=np.float32)
        padded[:arr.shape[0]] = arr
        res = torch.tensor(padded,dtype=torch.float)
        return res

    def _read(self, file_name,skew=0):
        lab_dict = {}
        with open(file_name, 'r') as file:
            content = file.readlines()
            lens = len(content)
            for idx in range(lens)[::2 + skew]:
                name = content[idx].replace('>', '').replace('\n', '')
                id,lig = name.split(' ')[0], name.split(' ')[1]
                lab = content[idx + 1 + skew].replace('\n', '')
                lab_dict[(id,lig)] = lab
        return lab_dict
    

class LoadData(Dataset): # 用于测试
    '''
        name_list: list of tuple, [(pdb_name,lig_name),...]
        proj_dir: str, path of label file
        lig_dict: 配体字典
        repr_dict: 归一化字典
    '''
    def __init__(self,name_list, proj_dir, lig_dict, repr_dict):
        self.name_list = name_list
        self.proj_dir     = proj_dir
        self.repr_dict    = repr_dict
        self.lig_dict = lig_dict
        
    def __len__(self):
        return len(self.name_list)
    def __getitem__(self, idx):
        name, lig = self.name_list[idx] # name是pdb_name
        feature_list = []
        feature_list.append(self.Normalize(np.load(f'{self.proj_dir}/dssp/{name}.npy'),self.repr_dict['dssp_max_repr'],self.repr_dict['dssp_min_repr']))
        feature_list.append(self.Normalize(np.load(f'{self.proj_dir}/ankh/{name}.npy'),self.repr_dict['ankh_max_repr'],self.repr_dict['ankh_min_repr']))
        feature  = np.concatenate(feature_list,axis=1) # rfeat
        xyz      = np.load(os.path.join(self.proj_dir,'pos',name+'.npy')) # xyz
        # ligand 信息
        ligand = self.Normalize(self.lig_dict[lig], self.repr_dict['ion_max_repr'], self.repr_dict['ion_min_repr'])# 768
        return name, lig, feature, ligand, xyz
    
    def Normalize(self, arr, max_value, min_value):
        scalar = max_value - min_value
        scalar[scalar==0] = 1
        return (arr - min_value) / scalar
    def collate_fn(self, batch):
        names, ligs, features, ligands, xyzs = zip(*batch)
        maxlen = 1500
        batch_rfeat = []
        batch_ligand = []
        batch_xyz = []
        batch_mask  = []
        for idx in range(len(batch)):
            batch_rfeat.append(self._padding(features[idx], maxlen)) # [ L, D]
            batch_ligand.append(torch.tensor(ligands[idx],dtype=torch.float))
            batch_xyz.append(self._padding(xyzs[idx], maxlen))
            mask = np.zeros(maxlen)
            mask[:features[idx].shape[0]] = 1
            batch_mask.append(torch.tensor(mask, dtype = torch.long))
        return names, ligs, torch.stack(batch_rfeat), torch.stack(batch_ligand), torch.stack(batch_xyz), torch.stack(batch_mask)
                  
    def _padding(self, arr, maxlen=1500):
        padded = np.zeros((maxlen,*arr.shape[1:]), dtype=np.float32)
        padded[:arr.shape[0]] = arr
        res = torch.tensor(padded,dtype=torch.float)
        return res