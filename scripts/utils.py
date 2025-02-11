import os
import numpy as np
import periodictable
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef, average_precision_score, roc_auc_score, roc_curve

def getRootPath():
    root_path = os.getcwd().replace('\\', '/').split('/')
    root_path = root_path[0: len(root_path)-1]
    root_path = '/'.join(root_path)
    return root_path

def getLigBind():
    lig_list = ['CA','MG','MN','ZN','FE','FE2','CU','NA','K','CO3','NO2','SO4','PO4','ADP','AMP','ATP','GDP','GTP','HEM']
    return lig_list

def getGPSite():
    gpsite_list = ['ATP','HEM','ZN','CA','MG','MN']
    return gpsite_list

def getMIonSite():
    ionic_list = ['CA', 'CD', 'CO', 'CU', 'FE', 'FE2', 'K', 'MG', 'MN', 'NA', 'NI','ZN']
    return ionic_list

def getNormProt():
    prot_list = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
                 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    return prot_list

def getNormALL():
    prot_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    return prot_list

def makeDir(path, name_list=[]):
    if not os.path.isdir(path):
        os.makedirs(path)

def appendText(path:str, text:str):
    with open(path, 'a') as file:
        file.write(text)
def writeText(path:str, text:str):
    with open(path, 'w') as file:
        file.write(text)
# ----
def lowerElem(elem):
    if len(elem) == 1:
        return elem
    return elem[0] + elem[1].lower()
def calMass(atom,pos=True):
    if pos:
        return periodictable.elements.symbol(lowerElem(atom.element)).mass * np.array(atom.get_coord())
    return periodictable.elements.symbol(lowerElem(atom.element)).mass

def mergeSeq(dest, target):
    dest = list(dest)
    target = list(target)
    return ''.join([str(int(x) or int(y)) for x, y in zip(dest, target)])

def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs

def readFasta(path,label,skew=0):
    res_dict = {}
    if label:
        with open(path, 'r') as file:
            content = file.readlines()
            lens = len(content)
            for idx in range(lens)[::2+skew]:
                name = content[idx].replace('>', '').replace('\n', '').strip()
                seq = content[idx + 1 + skew].replace('\n', '')
                if name in res_dict.keys():
                    res_dict[name] = mergeSeq(res_dict[name], seq)
                else:
                    res_dict[name] = seq
    else:
        with open(path, 'r') as file:
            content = file.readlines()
            lens = len(content)
            for idx in range(lens)[::2+skew]:
                name = content[idx].replace('>', '').replace('\n', '').strip()
                seq = content[idx + 1].replace('\n', '')
                res_dict[name] = seq
    return res_dict

def readDataList(path,skew=0):
    # only
    name_list= []
    with open(path, 'r') as file:
        content = file.readlines()
        lens = len(content)
        for idx in range(lens)[::2+skew]:
            name = content[idx].replace('>', '').replace('\n', '')
            if name.find(' ') == -1:
                id,lig = name,name
            else:
                id,lig = name.split(' ')[0], name.split(' ')[1]
            name_list.append((id,lig))
    return name_list

def getAbbr(name):
    res_dict = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'MSE': 'M', 'UNK': 'X', 'ASX': 'B', 'GLX': 'Z'}
    if name not in res_dict.keys():
        return 'X'
    return res_dict[name]

def lab2vec(text):
    if text == '':
        return []
    else:
        res_list = list(text)
        res_list = [int(i) for i in res_list]
    return np.array(res_list)

def calEval(y_true,y_score,save_path=None, best_th = 0.35):
    y_pred = [1 if i > best_th else 0 for i in y_score] # default, The optimal threshold needs to be selected through validation
    
    TN, FP, FN, TP = confusion_matrix(y_true,y_pred).ravel()
    
    if save_path != None:
        result = '\nRec: ' + str(recall_score(y_true,y_pred)) + '\n' + \
        'SPE: ' + str(TN/(TN+FP)) + '\n' + \
        'Acc: ' + str(accuracy_score(y_true,y_pred)) + '\n' + \
        'Pre: ' + str(precision_score(y_true,y_pred)) + '\n' + \
        'F1: ' + str(f1_score(y_true,y_pred)) + '\n' + \
        'MCC: ' + str(matthews_corrcoef(y_true,y_pred)) + '\n' + \
        'AUC: ' + str(roc_auc_score(y_true, y_score)) + '\n' + \
        'AUPR: ' + str(average_precision_score(y_true, y_score)) + '\n' 
        appendText(save_path,result)
        return
    else:
        return {'Rec':recall_score(y_true,y_pred),'SPE':TN/(TN+FP),'Acc':accuracy_score(y_true,y_pred),
                'Pre':precision_score(y_true,y_pred),'F1':f1_score(y_true,y_pred),
                'MCC':matthews_corrcoef(y_true,y_pred),'AUC':roc_auc_score(y_true, y_score),'AUPR':average_precision_score(y_true, y_score)}

def getBestThreshold(y_true,y_score):
    best_threshold = 0
    best_mcc = -1
    best_pred = []
    for i in range(100):
        threshold = i/100
        y_pred = [1 if i > threshold else 0 for i in y_score]
        mcc = matthews_corrcoef(y_true,y_pred)
        if mcc > best_mcc:
            best_mcc = mcc
            best_threshold = threshold
            best_pred = y_pred
    return best_threshold,best_mcc,best_pred
