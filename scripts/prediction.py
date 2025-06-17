import argparse
import gc
import os
import shutil
import pickle as pkl
import pandas as pd
import requests
import torch
from tqdm import tqdm
from utils import *
from scipy.spatial import cKDTree
from Bio.PDB.DSSP import DSSP
from Bio import SeqIO
from lxml import etree
from Bio.PDB.ResidueDepth import get_surface
from model import LABind
from readData import LoadData
from config import nn_config, pretrain_path
from torch.utils.data import DataLoader
from torch import nn
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, EsmForProteinFolding
from Bio.PDB import PDBParser
from download_weights import download_all_weights
# msms
def getMSMS(pdb_path,msms_path='../tools/msms'):
    """
    Args:
        pdb_file (str): 蛋白质结构文件路径
    """
    for pdb_file_name in tqdm(os.listdir(pdb_path),desc='MSMS running',ncols=80,unit='proteins'):
        pdb_file = pdb_path+pdb_file_name
        save_file = pdb_file.replace('.pdb','.npy').replace('pdb','pos')
        if os.path.exists(save_file):
            continue
        parser = PDBParser(QUIET=True)
        X = []
        chain_atom = ['N', 'CA', 'C', 'O']
        model = parser.get_structure('model', pdb_file)[0]
        chain = next(model.get_chains())
        try:
            surf = get_surface(chain,MSMS=msms_path)
            surf_tree = cKDTree(surf)
        except:
            surf = np.empty(0)
        for residue in chain:
            line = []
            atoms_coord = np.array([atom.get_coord() for atom in residue])
            if surf.size == 0:
                dist, _ = surf_tree.query(atoms_coord)
                closest_atom = np.argmin(dist)
                closest_pos = atoms_coord[closest_atom]
            else:
                closest_pos = atoms_coord[-1]
            atoms = list(residue.get_atoms())
            ca_pos= residue['CA'].get_coord() if 'CA' in residue else residue.child_list[0].get_coord()
            pos_s = 0
            un_s = 0
            for atom in atoms:
                if atom.name in chain_atom:
                    line.append(atom.get_coord())
                else:
                    pos_s += calMass(atom,True)
                    un_s += calMass(atom,False)
            # 此处line应该等于4
            if len(line) != 4:
                line = line + [list(ca_pos)]*(4-len(line))
            if un_s == 0:
                R_pos = ca_pos
            else:
                R_pos = pos_s / un_s
            line.append(R_pos)  
            line.append(closest_pos) # 加入最近点的残基信息
            X.append(line) 
        np.save(save_file, X)

# dssp
def getDSSP(pdb_path,dssp_path='../tools/dssp'): # get dssp feature
    mapSS = {' ':[0,0,0,0,0,0,0,0,0],
             '-':[1,0,0,0,0,0,0,0,0],
             'H':[0,1,0,0,0,0,0,0,0],
             'B':[0,0,1,0,0,0,0,0,0],
             'E':[0,0,0,1,0,0,0,0,0],
             'G':[0,0,0,0,1,0,0,0,0],
             'I':[0,0,0,0,0,1,0,0,0],
             'P':[0,0,0,0,0,0,1,0,0],
             'T':[0,0,0,0,0,0,0,1,0],
             'S':[0,0,0,0,0,0,0,0,1]}
    p = PDBParser(QUIET=True)
    for pdb_file_name in tqdm(os.listdir(pdb_path),desc='DSSP running',ncols=80,unit='proteins'):
        pdb_file = pdb_path+pdb_file_name
        save_file = pdb_file.replace('.pdb','.npy').replace('pdb','dssp')
        if os.path.exists(save_file):
            continue
        structure = p.get_structure("tmp", pdb_file)
        model = structure[0]
        try:
            dssp = DSSP(model, pdb_file, dssp=dssp_path)
            keys = list(dssp.keys())
        except:
            keys = []
        res_np = []
        for chain in model:
            for residue in chain:
                res_key = (chain.id,(' ', residue.id[1], residue.id[2]))
                if res_key in keys:
                    tuple_dssp = dssp[res_key]
                    res_np.append(mapSS[tuple_dssp[2]] + list(tuple_dssp[3:]))
                else:
                    res_np.append(np.zeros(20))
        res_data = np.array(res_np)
        if res_data.dtype == '<U32':
            res_data = np.where(res_data == 'NA', 0, res_data).astype(np.float32)
        np.save(save_file, np.array(res_np))

# getEmbed
def getEmbed(fasta_file,embed_path='../tools/pretrain/ankh',device='cuda',out_path='./'):
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    if len(sequences) == len(os.listdir(out_path)):
        print('The number of proteins is consistent with the number of Ankh files, so Ankh prediction is not required.')
        return
    # 使用ankh对蛋白质进行特征提取
    tokenizer = AutoTokenizer.from_pretrained(embed_path)
    model     = T5EncoderModel.from_pretrained(embed_path)
    model.to(device)
    model.eval()
    for record in tqdm(sequences, desc='Ankh running',ncols=80,unit='proteins'):
        if os.path.exists(out_path+f'{record.id}.npy'):
            continue
        ids = tokenizer.batch_encode_plus([list(record.seq)], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
        input_ids = ids['input_ids'].to(device)
        attention_mask = ids['attention_mask'].to(device)
        with torch.no_grad():
            embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)
            emb = embedding_repr.last_hidden_state[0,:len(record.seq)].cpu().numpy()
            np.save(out_path+f'{record.id}.npy',emb)
    del model
    gc.collect()
    
def downloadLigand(name):
    pdb_url = f'https://www.rcsb.org/ligand/{name}'
    response = requests.get(pdb_url)
    # 使用xpath获取信息
    html_content = response.content
    tree = etree.HTML(html_content)
    smiles= tree.xpath('//tr[@id="chemicalIsomeric"]/td/text()')[0]
    return smiles

def getMolEmbed(fasta_file,smiles_file=None,mol_path='../tools/pretrain/molformer',device='cuda'):
    if os.path.exists(smiles_file):
        with open(smiles_file,'r') as f:
            smiles_dict = {line.split()[0]:line.split()[1] for line in f}
        if os.path.abspath(smiles_file) != os.path.abspath(fasta_file.replace('protein.fa','smiles.txt')):
            shutil.copy(smiles_file,fasta_file.replace('protein.fa','smiles.txt'))
    else:
        name_list = readDataList(fasta_file) 
        # 去重复
        lig_list = list(set([lig for _,lig in name_list]))
        smiles_dict = {}
        smiles_str = ''
        for lig_name in tqdm(lig_list,desc='downloading smiles from RCSB',ncols=80,unit='molecules'):
            smiles = downloadLigand(lig_name)
            smiles_dict[lig_name] = smiles
            smiles_str += f'{lig_name} {smiles}\n'
        writeText(fasta_file.replace('protein.fa','smiles.txt'),smiles_str)
        
    model = AutoModel.from_pretrained(mol_path, deterministic_eval=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(mol_path, trust_remote_code=True)
    model.to(device)
    res_dict = {}
    for smiles_name in tqdm(smiles_dict,desc='MolFormer running',ncols=80,unit='molecules'):
        smiles = smiles_dict[smiles_name]
        with torch.no_grad():
            inputs = tokenizer(smiles, padding=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            pooler = outputs.pooler_output.cpu().numpy()
            res_dict[smiles_name] = pooler
    # 保存文件
    with open(fasta_file.replace('protein.fa','ligand.pkl'), 'wb') as f:
        pkl.dump(res_dict, f)
    del model
    gc.collect()
    
def getESMFold(fasta_file,fold_path='../tools/pretrain/esmfold_v1/', out_path='./', device='cuda'):
    # 提前判断
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    flag = True
    for record in sequences:
        if not os.path.exists(out_path+record.id+'.pdb'):
            flag = False
            break
    if flag:
        print('The number of proteins is consistent with the number of pdb files, so ESMFold prediction is not required.')
        return
    # 如果不存在则创建
    tokenizer = AutoTokenizer.from_pretrained(fold_path)
    model = EsmForProteinFolding.from_pretrained(fold_path, low_cpu_mem_usage=True)
    model = model.eval()
    model.esm = model.esm.half()
    model = model.to(device)
    
    for record in tqdm(sequences, desc='ESMFold running', ncols=80, unit='proteins'):
        if os.path.exists(out_path+record.id+'.pdb'):
            continue
        tokenized_input = tokenizer([str(record.seq)], return_tensors="pt", add_special_tokens=False)['input_ids']
        tokenized_input = tokenized_input.to(device)
        with torch.no_grad():
            output = model(tokenized_input)
        pdb = convert_outputs_to_pdb(output)
        with open(out_path+record.id+'.pdb', "w") as f:
            f.write(''.join(pdb))
    del model
    gc.collect()
    
def getReprDict(path, repr_path):
    with open(path+'/ligand.pkl','rb') as f:# 配体嵌入
        ligand_dict = pkl.load(f)
    with open(repr_path,'rb') as f:
        repr_dict = pkl.load(f)
    return ligand_dict, repr_dict  
  
def prediction(fasta_path,batch_size=5,device_ids=[0],out_path='./',model_path='../model/Unseen/'):
    run_device = f'cuda:{device_ids[0]}'
    # 加载模型
    models = []
    for fold in range(5):
        state_dict = torch.load(model_path + 'fold%s.ckpt'%fold, run_device)
        model = LABind(
        rfeat_dim=nn_config['rfeat_dim'], ligand_dim=nn_config['ligand_dim'], hidden_dim=nn_config['hidden_dim'], heads=nn_config['heads'], augment_eps=nn_config['augment_eps'], 
        rbf_num=nn_config['rbf_num'],top_k=nn_config['top_k'], attn_drop=nn_config['attn_drop'], dropout=nn_config['dropout'], num_layers=nn_config['num_layers']).to(run_device)
        model = nn.DataParallel(model,device_ids=device_ids)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    ligand_dict, repr_dict = getReprDict(out_path,repr_path='../tools/repr.pkl')
    fa_dict = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
    with open(out_path+'/smiles.txt','r') as f:
        smiles_dict = {line.split()[0]:line.split()[1] for line in f}
        
    test_list = readDataList(fasta_path)
    test_data = LoadData(name_list=test_list,
                         proj_dir=out_path,
                         lig_dict=ligand_dict,
                         repr_dict=repr_dict)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=test_data.collate_fn,shuffle=False,drop_last=False)
    
    df = pd.DataFrame(columns=['Protein Name','Protein Sequence','Ligand Name','Ligand SMILES','Binding Site Probability'])
    with torch.no_grad():
        for names, ligs, rfeat, ligand, xyz, mask in tqdm(test_loader, desc='LABind running', ncols=80, unit='batches'):
            tensors = [rfeat, ligand, xyz,  mask]
            tensors = [tensor.to(run_device) for tensor in tensors]
            rfeat, ligand, xyz, mask = tensors
            logits = [model(rfeat, ligand, xyz, mask).sigmoid() for model in models]
            logits = torch.stack(logits,0).mean(0)
            logits = logits.half()
            logits = logits.cpu().detach().numpy()
            # 转换为float64
            for idx,logit in enumerate(logits):
                df = pd.concat([df,pd.DataFrame([[names[idx], str(fa_dict[names[idx]].seq), 
                ligs[idx], smiles_dict[ligs[idx]],list(logit[:mask[idx].sum()])]],columns=df.columns)])
    # 按顺序保存为csv文件
    df.to_csv(out_path+'/RESULT.csv',index=False)

                
def SetParser(parser_args):
    # 如果不存在预训练的模型，则下载到config的模型路径
    download_all_weights(pretrain_path = pretrain_path)
    
    gpus = parser_args.gpu_id
    run_device = f'cuda:{gpus[0]}' if torch.cuda.is_available() else 'cpu'
    # # 创建输出文件夹
    os.makedirs(parser_args.outpath, exist_ok = True), os.makedirs(parser_args.outpath+'/pdb/', exist_ok = True), os.makedirs(parser_args.outpath+'/ankh', exist_ok = True)
    os.makedirs(parser_args.outpath+'/dssp/', exist_ok = True), os.makedirs(parser_args.outpath+'/pos/', exist_ok = True)
    # 判断pdbpath是否存在
    if parser_args.input_pdbpath:
        if not os.path.exists(parser_args.input_pdbpath):
            raise FileNotFoundError(f'pdb path {parser_args.input_pdbpath} does not exist!')
        # 从pdb路径提取fasta文件中有的蛋白质
        name_list = readDataList(parser_args.input_fasta)
        for prot,_ in name_list:
            if os.path.exists(parser_args.input_pdbpath+f'/{prot}.pdb') and os.path.abspath(parser_args.input_pdbpath+f'/{prot}.pdb') != os.path.abspath(parser_args.outpath+'/pdb/'+f'{prot}.pdb'):
                shutil.copy(parser_args.input_pdbpath+f'/{prot}.pdb',parser_args.outpath+'/pdb/')
    #使用esmfold生成pdb文件 or 生成没有pdb的文件
    getESMFold(parser_args.input_fasta, fold_path=pretrain_path['esmfold_path'], out_path=parser_args.outpath+'/pdb/', device=run_device)
    parser_args.input_pdbpath = parser_args.outpath+'/pdb/'
    
    # # 提取DSSP和MSMS
    getDSSP(parser_args.input_pdbpath,dssp_path='../tools/mkdssp')
    getMSMS(parser_args.input_pdbpath,msms_path='../tools/msms')
    if os.path.abspath(parser_args.input_fasta) != os.path.abspath(parser_args.outpath+'/protein.fa'):
        shutil.copy(parser_args.input_fasta, parser_args.outpath+'/protein.fa')
    parser_args.input_fasta = parser_args.outpath+'/protein.fa'
    
    # 使用ankh生成embed文件
    getEmbed(parser_args.input_fasta,embed_path=pretrain_path['ankh_path'], device=run_device, out_path=parser_args.outpath+'/ankh/')
    getMolEmbed(parser_args.input_fasta, smiles_file=parser_args.input_ligand, mol_path=pretrain_path['molformer_path'], device=run_device)
    
    prediction(parser_args.input_fasta,batch_size=parser_args.batch, device_ids=gpus, out_path=parser_args.outpath)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='predict protein ligand binding site, you can input PDB file and protein sequence! Good Luck!')
    
    parser.add_argument("-if", "--input_fasta", type = str, help = "Input fasta file, if you enter a fasta file, you are not allowed to enter a protein structure", required=True)
    parser.add_argument("-ip", "--input_pdbpath", type = str, help = "Input a pdb path, proteins that have PDB files will no longer use ESMFold for prediction.")
    parser.add_argument("-il", "--input_ligand", type= str, default='', help = "Input the lignad smiles file, like prot_name lig_name. Please use the RCSB standard ligand representation.")
    
    parser.add_argument("-op", "--outpath", type = str, default='./output/',help = "Output path to save intermediate files and final predictions") #保存对于配体的结合概率

    parser.add_argument("-b", "--batch", type = int, default = 1, help = "Batch size of the model prediction")
    parser.add_argument("-g", "--gpu_id", default = [0], nargs='+', help = "The GPU id used for feature extraction and binding site prediction, can parallel!")
    
    args = parser.parse_args()
    args = SetParser(args)