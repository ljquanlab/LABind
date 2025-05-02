# download pretrained weights for LABind
import os
from huggingface_hub import snapshot_download
pretrain_path = { # Please modify 
    'esmfold_path': '../tools/esmfold_v1', # esmfold path
    'ankh_path': '../tools/ankh-large/', # ankh path
    'molformer_path': '../tools/MoLFormer-XL-both-10pct/', # molformer path
}
if 'HF_MIRROR' in os.environ: # set huggingface mirror
    HF_MIRROR = os.environ['HF_MIRROR']
else:
    HF_MIRROR = "https://huggingface.co/" # If you're in China, please set https://hf-mirror.com/ 

# download pretrained weights
def download_all_weights(outpath= None, pretrain_path = None):
    # Ankh
    snapshot_download(
        repo_id = "ElnaggarLab/ankh-large", 
        local_dir = outpath + "/ankh-large" if pretrain_path == None else pretrain_path["ankh_path"],
        allow_patterns= "*",
        endpoint = HF_MIRROR
    )
    # ESMFold
    snapshot_download(
        repo_id = "facebook/esmfold_v1", 
        local_dir = outpath + "/esmfold_v1" if pretrain_path == None else pretrain_path["esmfold_path"],
        allow_patterns= "*",
        endpoint = HF_MIRROR
    )
    # MolFormer
    snapshot_download(
        repo_id = "ibm/MoLFormer-XL-both-10pct", 
        local_dir = outpath + "/MoLFormer-XL-both-10pct" if pretrain_path == None else pretrain_path["molformer_path"],
        allow_patterns= "*",
        endpoint = HF_MIRROR
    )
    print("All models downloaded successfully.")
    

if __name__ == "__main__":
    # set up argument parser
    import argparse
    parser = argparse.ArgumentParser(description='Download pretrained weights for LABind')
    parser.add_argument("-o", "--outpath", type=str, default='../tools/', help='Saved path')
    parser.add_argument("-e", "--endpoint", type=str, default=HF_MIRROR, help='Huggingface endpoint')
    args = parser.parse_args()
    outpath = args.outpath
    HF_MIRROR = args.endpoint
    # download pretrained weights
    download_all_weights(outpath)