
# download BioLiP.txt
import argparse
import os
import re
import shutil
import subprocess

from tqdm import tqdm
from utils import readFasta,appendText, writeText

def download_biolip(out_path):
    url = "https://zhanggroup.org/BioLiP/download/BioLiP.txt.gz"
    out_path = os.path.join(out_path, "BioLiP.txt.gz")
    if not os.path.exists(out_path):
        os.system(f"wget {url} -O {out_path}")
    # unzip
    os.system(f"gunzip {out_path}")
    return out_path[:-3]
    
def bio2fasta(biolip_path, out_path, max_length):
    res_str = ""
    with open(biolip_path, "r") as f:
        for line in tqdm(f):
            item = line.strip().split("\t")
            length = len(item[-1].strip())
            if item[2]!='' and float(item[2]) <= 3.0 and float(item[2]) > 0 and length <= max_length: # 根据需要进行修改
                res_str += f">{item[0]}{item[1]} {item[4]}\n{item[-1]}\n"
                pos = item[8]
                pos_num = re.sub(r'[A-Z]|[a-z]', '', pos).split('\40')
                label_seq = '0' * len(item[-1].replace('\n', ''))
                label_seq = list(label_seq)
                for idx in pos_num:
                    label_seq[int(idx) - 1] = '1'
                label_seq = ''.join(label_seq)
                res_str += f"{label_seq}\n"
    writeText(out_path+"/biolip.fa",res_str)
    return out_path+"/biolip.fa"

def merge_fasta(fasta_path, out_path):
    out = f"{out_path}/"
    fa_dict = readFasta(fasta_path,label=False,skew=1)
    la_dict = readFasta(fasta_path,label=True,skew=1)
    res_str = ""
    for key in tqdm(fa_dict):
        res_str += f">{key}\n{fa_dict[key]}\n{la_dict[key]}\n"
    writeText(out+"biolip_merge.fa",res_str)

def split_fasta(fasta_path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    fa_dict = readFasta(fasta_path,label=False,skew=1)
    la_dict = readFasta(fasta_path,label=True,skew=1)
    res_dict = {}
    for key in tqdm(fa_dict):
        ligand_id = key.split(' ')[1]
        if ligand_id not in res_dict:
            res_dict[ligand_id] = f">{key}\n{fa_dict[key]}\n{la_dict[key]}\n"
        else:
            res_dict[ligand_id] += f">{key}\n{fa_dict[key]}\n{la_dict[key]}\n"
            
    for key in res_dict:
        writeText(f"{out_path}/{key}.fa",f"{res_dict[key]}")

def split_seq(path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for file in tqdm(os.listdir(path)):
        fa_dict = readFasta(f"{path}/{file}",label=False,skew=1)
        res_str = ""
        if len(fa_dict) <2:
            continue
        for key in fa_dict:
            res_str += f">{key}\n{fa_dict[key]}\n"
        writeText(f"{out_path}/{file}",res_str)

def cluster(path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for file in tqdm(os.listdir(path)):
        command = f"mmseqs easy-cluster {path}/{file} {out_path}/{file} tmp/mmseqs/ --min-seq-id 0.3 -c 0.3 --cov-mode 1"
        subprocess.run(command, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        # remove file
        os.remove(f"{out_path}/{file}_all_seqs.fasta")
        os.remove(f"{out_path}/{file}_cluster.tsv")
        os.rename(f"{out_path}/{file}_rep_seq.fasta", f"{out_path}/{file}")

def setLabel(label_path, path, out_path):
    # 移除小于2个样本的文件
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        
    for file in os.listdir(path):
        if len(readFasta(f"{path}/{file}",label=False)) < 2:
            os.remove(f"{path}/{file}")
            
    for file in os.listdir(path):
        fa_dict = readFasta(f"{path}/{file}",label=False)
        la_dict = readFasta(f"{label_path}/{file}",label=True,skew=1)
        res_str = ""
        for key in fa_dict:
            res_str += f">{key}\n{fa_dict[key]}\n{la_dict[key]}\n"
        writeText(f"{out_path}/{file}",res_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_path", type=str, default="data/")
    parser.add_argument("-m", "--max_length", type=int, default=1500)
    args = parser.parse_args()
    out_path = args.out_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    download_biolip(out_path)
    bio2fasta(out_path+"/BioLiP.txt", out_path, args.max_length)
    merge_fasta(out_path+"/biolip.fa", out_path)
    split_fasta(out_path+"/biolip_merge.fa", out_path+"/ligand/")
    split_seq(out_path+"/ligand/", out_path+"/origin_fasta/")
    cluster(out_path+"/origin_fasta/", out_path+"/fasta/")
    setLabel(out_path+"/ligand/", out_path+"/fasta/", out_path+"/label/")
    print("Done!")
    