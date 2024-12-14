# train.py
import datetime
from torch.utils.data import DataLoader
from readData import readData
from utils import *
import torch
from torch import nn
from model import LABind
from func_help import setALlSeed,get_std_opt
from config import nn_config
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
import wandb
import random
root_path = getRootPath()
setALlSeed(11)
wandb.login()
sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'AUPR',
        'goal': 'maximize'
    },
    'parameters' :{}
}

# 定义搜索参数
sweep_config['parameters'].update({ # 关键参数搜索
    'hidden_dim': {
        'values': [256] # [32,64,128,256] 
    },
    'heads': {
        'values': [4] # [4,6,8,10] 
    },
    'augment_eps': {
        'values': [0.1]# [0, 0.05, 0.1, 0.2, 0.3] 
    },
    'rbf_num': {
        'values': [8] # [8, 16, 32, 64] # [8] 
    },
    'top_k': {
        'values': [30]# [10, 20, 30, 40] 
    },
    'attn_drop': {
        'values': [0.1] # [0.2]
    },
    'dropout': {
        'values': [0.3]# [0.2] # [0.1, 0.2, 0.3, 0.4]
    },
    'num_layers': {
        'values': [4] # [1, 2, 3, 4] 
    },
    'learning_rate': {
        'values': [4e-4] # [1e-4,3e-4,5e-4] 
    },
})

DEVICE = 'cuda:0'

def valid(model, valid_list):
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
        return aupr_value

def test(models, test_list):
    test_data = readData(
        name_list=test_list, 
        proj_dir=nn_config['proj_dir'], 
        lig_dict=nn_config['lig_dict'],
        true_file=nn_config['test_file'])
    # 打印长度
    test_loader = DataLoader(test_data, batch_size=nn_config['batch_size'], collate_fn=test_data.collate_fn)
    all_y_score = []
    all_y_true = []
    with torch.no_grad():
        for rfeat, ligand, xyz,  mask, y_true in test_loader:
            tensors = [rfeat, ligand, xyz,  mask, y_true]
            tensors = [tensor.to(DEVICE) for tensor in tensors]
            rfeat, ligand, xyz, mask, y_true = tensors
            logits = [model(rfeat, ligand, xyz, mask).sigmoid() for model in models]
            logits = torch.stack(logits,1).mean(1)
            # CA CD CO CU FE FE2 K MG MN NA NI ZN all
            # 多任务学习
            # 计算所有离子的loss
            logits = torch.masked_select(logits, mask==1)
            y_true = torch.masked_select(y_true, mask==1)
            all_y_score.extend(logits.cpu().detach().numpy())
            all_y_true.extend(y_true.cpu().detach().numpy())
    return average_precision_score(all_y_true, all_y_score)

# 创建数据集----------------
kf = KFold(n_splits=5, shuffle=True, random_state = 42)
data_list = readDataList(f'{root_path}/GPSite/label/train/train.fa',skew=1)
# ------------------------

# train
def train(config=None):
    
    with wandb.init(config=config):
        config = wandb.config
    loss_fn = nn.BCELoss(reduction='none')
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project='LABind', config = config.__dict__, name = nowtime, save_code=False)
    # 进行5折交叉验证 or 不进行5折加快速度
    fold_idx = 0
    for train_idx, valid_idx in kf.split(data_list):
        train_list = [data_list[idx] for idx in train_idx]
        valid_list = [data_list[idx] for idx in valid_idx]
        train_data = readData(
            name_list=train_list, 
            proj_dir=nn_config['proj_dir'], 
            lig_dict=nn_config['lig_dict'],
            true_file=nn_config['train_file'])
        train_loader = DataLoader(train_data, batch_size=nn_config['batch_size'],shuffle=True, collate_fn=train_data.collate_fn, num_workers=5)
        model = LABind(rfeat_dim=nn_config['rfeat_dim'], ligand_dim=nn_config['ligand_dim'],
                    hidden_dim=config.hidden_dim, heads=config.heads, augment_eps=config.augment_eps, 
                    rbf_num=config.rbf_num, top_k=config.top_k, attn_drop=config.attn_drop, dropout=config.dropout, num_layers=config.num_layers)
        model.to(DEVICE)
        model = nn.DataParallel(model, device_ids=nn_config['device_ids'])
        optimizer = get_std_opt(len(train_data),nn_config['batch_size'], model.parameters(), config.hidden_dim, config.learning_rate)
        v_max_aupr = 0
        patience = 0
        for _ in range(70):
            all_loss = 0
            all_cnt = 0
            model.train()
            for rfeat, ligand, xyz,  mask, y_true in tqdm(train_loader):
                tensors = [rfeat, ligand, xyz,  mask, y_true]
                tensors = [tensor.to(DEVICE) for tensor in tensors]
                rfeat, ligand, xyz, mask, y_true = tensors
                optimizer.zero_grad()
                logits = model(rfeat, ligand, xyz,  mask).sigmoid() # [N]
                # 计算所有离子的loss
                loss = loss_fn(logits, y_true) * mask
                loss = loss.sum() / mask.sum()
                all_loss += loss.item()
                all_cnt += 1
                loss.backward()
                optimizer.step()
            epoch_loss = all_loss / all_cnt
            v_aupr = valid(model,valid_list) # 计算验证集的aupr
            wandb.log({f'loss': epoch_loss, f'valid_aupr':v_aupr})
            if v_aupr > v_max_aupr:
                v_max_aupr = v_aupr
                patience = 0
                torch.save(model.state_dict(), f'{root_path}/Output/WandB/fold{fold_idx}.ckpt')
            else:
                patience += 1
            if patience >= nn_config['max_patience']:
                break
        fold_idx += 1
    # 训练结束
    
    # 计算测试集的MCC
    models = []
    for test_fold_idx in range(5):
        model = LABind(rfeat_dim=nn_config['rfeat_dim'], ligand_dim=nn_config['ligand_dim'],
                    hidden_dim=config.hidden_dim, heads=config.heads, augment_eps=config.augment_eps, 
                    rbf_num=config.rbf_num, top_k=config.top_k, attn_drop=config.attn_drop, dropout=config.dropout, num_layers=config.num_layers)
        model.to(DEVICE)
        model = nn.DataParallel(model, device_ids=nn_config['device_ids'])
        model.load_state_dict(torch.load(f'{root_path}/Output/WandB/fold{test_fold_idx}.ckpt',DEVICE))
        model.eval()
        models.append(model)
    
    test_table = wandb.Table(columns=["ligand","AUPR"])
    avg_aupr = 0
    with torch.no_grad():
        for test_ionic in getGPSite():
            test_ionic_list = readDataList(f'{root_path}/GPSite/label/test/{test_ionic}.fa',skew=1)
            aupr = test(models, test_ionic_list)
            test_table.add_data(test_ionic,aupr)
            avg_aupr += aupr
    wandb.log({'test':test_table})
    wandb.log({'AUPR':avg_aupr/len(getGPSite())})
    wandb.finish()
    return
# valid

if __name__ == '__main__':
    sweep_id = wandb.sweep(sweep_config, project='LABind')
    wandb.agent(sweep_id, function=train,count=4)