import torch
from utils import *
from model import LABind
from readData import readData
from config import nn_config
from torch.utils.data import DataLoader
from torch import nn
from config import dataset
import pickle as pkl
model_class = dataset
root_path = getRootPath()

def predict(models, device):
    df = pd.DataFrame(columns=['ligand','Rec','SPE','Acc','Pre','F1','MCC','AUC','AUPR'])
    for ionic in getLigBind():
        test_list = readDataList(f'{root_path}/{dataset}/label/test/{ionic}.fa',skew=1)
        test_data = readData(
            name_list=test_list, 
            proj_dir=nn_config['proj_dir'], 
            lig_dict=nn_config['lig_dict'],
            true_file=f'{root_path}/{dataset}/label/test/test.fa')
        # 打印长度
        test_loader = DataLoader(test_data, batch_size=nn_config['batch_size'], collate_fn=test_data.collate_fn)
        print(f'{ionic} test data length: {len(test_data)}')
        all_y_score = []
        all_y_true = []
        with torch.no_grad():
            for rfeat, ligand, xyz,  mask, y_true in test_loader:
                tensors = [rfeat, ligand, xyz,  mask, y_true]
                tensors = [tensor.to(device) for tensor in tensors]
                rfeat, ligand, xyz, mask, y_true = tensors
                logits = [model(rfeat, ligand, xyz, mask).sigmoid() for model in models] # [B, N, 256]
                logits = torch.stack(logits,0).mean(0)
                logits = torch.masked_select(logits, mask==1)
                y_true = torch.masked_select(y_true, mask==1)
                all_y_score.extend(logits.cpu().detach().numpy())
                all_y_true.extend(y_true.cpu().detach().numpy())
        data_dict = calEval(all_y_true, all_y_score,f'{model_path}/{ionic}.log')
        data_dict['ligand'] = ionic
        df = pd.concat([df,pd.DataFrame(data_dict,index=[0])])
    df.to_csv(f'{model_path}test.csv',index=False)
if __name__ == '__main__':
    device = 'cuda:0'
    model_path = f'{root_path}/Output/{model_class}_Default/'
    models = []
    print(model_path)
    print(nn_config['pdb_class'])
    for fold in range(5): # 5-fold avg
        state_dict = torch.load(model_path + 'fold%s.ckpt'%fold,'cuda:0')
        model = LABind(
        rfeat_dim=nn_config['rfeat_dim'], ligand_dim=nn_config['ligand_dim'], hidden_dim=nn_config['hidden_dim'], heads=nn_config['heads'], augment_eps=nn_config['augment_eps'], 
        rbf_num=nn_config['rbf_num'],top_k=nn_config['top_k'], attn_drop=nn_config['attn_drop'], dropout=nn_config['dropout'], num_layers=nn_config['num_layers']).to(device)
        model = nn.DataParallel(model,device_ids=nn_config['device_ids'])
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    predict(models,'cuda:0')
        