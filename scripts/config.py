from utils import getRootPath
import numpy as np
import pickle as pkl
root_path = getRootPath()

dataset = 'LigBind'

nn_config = {
    # dataset 
    'train_file': f'{root_path}/{dataset}/label/train/train.fa',
    'test_file': f'{root_path}/{dataset}/label/test/test.fa',
    'never_file': f'{root_path}/{dataset}/label/never/never.fa',
    'proj_dir': f'{root_path}/{dataset}/',
    'lig_dict': pkl.load(open(f'{root_path}/tools/ligand.pkl', 'rb')),
    'pdb_class':'source',
    'dssp_max_repr': np.load(f'{root_path}/tools/dssp_max_repr.npy'),
    'dssp_min_repr': np.load(f'{root_path}/tools/dssp_min_repr.npy'),
    'ankh_max_repr': np.load(f'{root_path}/tools/ankh_max_repr.npy'),
    'ankh_min_repr': np.load(f'{root_path}/tools/ankh_min_repr.npy'),
    'ion_max_repr': np.load(f'{root_path}/tools/ion_max_repr.npy'),
    'ion_min_repr': np.load(f'{root_path}/tools/ion_min_repr.npy'),
    # model parameters
    
    'rfeat_dim':1556,
    'ligand_dim':768, 
    'hidden_dim':256, 
    'heads':4, 
    'augment_eps':0.1, 
    'rbf_num':8, 
    'top_k':30, 
    'attn_drop':0.1, 
    'dropout':0.3, 
    'num_layers':4, 
    'lr':0.0004, 
    
    # training parameters
    'batch_size':15,
    'max_patience':10,
    'device_ids':[5,3,7,8,9],
}
pretrain_path = { # Please modify 
    'esmfold_path': '/home/ylujiang/zzjun/tools/esmfold_v1', # esmfold path
    'ankh_path': '/home/ylujiang/zzjun/tools/ankh-large/', # ankh path
    'molformer_path': '/home/ylujiang/zzjun/tools/MoLFormer-XL-both-10pct/', # molformer path
    'model_path':f'{root_path}/model/Unseen/' # based on Unseen
}
