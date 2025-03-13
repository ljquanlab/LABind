from torch.utils.data import DataLoader
from readData import readData
from utils import *
import torch
from torch import nn
from model import LABind
from func_help import setALlSeed,get_std_opt
from config import nn_config
from config import dataset
from tqdm import tqdm
from sklearn.model_selection import KFold
DEVICE = torch.device('cuda:0')
root_path = getRootPath()

def valid(model, valid_list,fold_idx=None):
    model.to(DEVICE)
    model.eval()
    valid_data = readData(
        name_list=valid_list, 
        proj_dir=nn_config['proj_dir'], 
        lig_dict=nn_config['lig_dict'],
        true_file=nn_config['train_file'])
    valid_loader = DataLoader(valid_data, batch_size=nn_config['batch_size'],shuffle=True, collate_fn=valid_data.collate_fn, num_workers=5)
    all_y_score = []
    all_y_true = []
    with torch.no_grad():
        for rfeat, ligand, xyz,  mask, y_true in valid_loader:
            tensors = [rfeat, ligand, xyz,  mask, y_true]
            tensors = [tensor.to(DEVICE) for tensor in tensors]
            rfeat, ligand, xyz, mask, y_true = tensors
            logits = model(rfeat, ligand, xyz,  mask).sigmoid() # [N]
            logits = torch.masked_select(logits, mask==1)
            y_true = torch.masked_select(y_true, mask==1)
            all_y_score.extend(logits.cpu().detach().numpy())
            all_y_true.extend(y_true.cpu().detach().numpy())
        # 通过aupr数值进行早停
        aupr_value = average_precision_score(all_y_true, all_y_score)
        appendText(f'{root_path}/Output/{dataset}/valid.log', f'valid {fold_idx}: {aupr_value}\n')
    return aupr_value

def taskTrain(train_list,valid_list=None,model=None,epochs=50,fold_idx=None):
    model.to(DEVICE)
    train_data = readData(
        name_list=train_list, 
        proj_dir=nn_config['proj_dir'], 
        lig_dict=nn_config['lig_dict'],
        true_file=nn_config['train_file'])
    train_loader = DataLoader(train_data, batch_size=nn_config['batch_size'],shuffle=True, collate_fn=train_data.collate_fn, num_workers=5)
    loss_fn = nn.BCELoss(reduction='none')
    # loss_fn = FocalLoss(alpha=0.75,gamma=1)
    optimizer = get_std_opt(len(train_list),nn_config['batch_size'], model.parameters(), nn_config['hidden_dim'], nn_config['lr'])
    v_max_aupr = 0
    patience = 0
    t_mccs = []
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model,device_ids=nn_config['device_ids'])
    train_losses = []
    for epoch in range(epochs):
        all_loss = 0
        all_cnt = 0
        model.train()
        for rfeat, ligand, xyz,  mask, y_true in tqdm(train_loader):
            tensors = [rfeat, ligand, xyz,  mask, y_true]
            tensors = [tensor.to(DEVICE) for tensor in tensors]
            rfeat, ligand, xyz, mask, y_true = tensors
            optimizer.zero_grad()
            logits = model(rfeat, ligand, xyz, mask).sigmoid() # [N]
            # 计算所有离子的loss
            loss = loss_fn(logits, y_true) * mask
            loss = loss.sum() / mask.sum()
            all_loss += loss.item()
            all_cnt += 1
            loss.backward()
            optimizer.step()
        train_losses.append(all_loss / all_cnt)
        # 根据验证集的aupr进行早停
        if valid_list is not None:
            v_aupr = valid(model,valid_list,fold_idx)
            t_mccs.append(v_aupr)
            print(f'Epoch {epoch} Loss: {all_loss / all_cnt}', f'Epoch Valid {epoch} AUPR: {v_aupr}')
            if v_aupr > v_max_aupr:
                v_max_aupr = v_aupr
                patience = 0
                torch.save(model.state_dict(), f'{root_path}/Output/{dataset}/fold{fold_idx}.ckpt')
            else:
                patience += 1
            if patience >= nn_config['max_patience']:
                break
    return 
    
def fold5train():
    print(f'-------------{dataset}-----------------')
    print('-------------5-fold cross validation-----------------')
    setALlSeed(11)
    kf = KFold(n_splits=5, shuffle=True, random_state = 42)
    data_list = readDataList(f'{root_path}/{dataset}/label/train/train.fa',skew=1)
    # data_list = BalanceData(data_list)
    fold_idx = 0
    for train_idx, valid_idx in kf.split(data_list):
        train_list = [data_list[i] for i in train_idx]
        valid_list = [data_list[j] for j in valid_idx]
        model = LABind(
        rfeat_dim=nn_config['rfeat_dim'], ligand_dim=nn_config['ligand_dim'], hidden_dim=nn_config['hidden_dim'], heads=nn_config['heads'], augment_eps=nn_config['augment_eps'], 
        rbf_num=nn_config['rbf_num'],top_k=nn_config['top_k'], attn_drop=nn_config['attn_drop'], dropout=nn_config['dropout'], num_layers=nn_config['num_layers'])
        taskTrain(train_list,valid_list,model,epochs=70,fold_idx=fold_idx)
        fold_idx += 1

if __name__ == '__main__':
    os.makedirs(f'{root_path}/Output/{dataset}',exist_ok=True)
    fold5train()
