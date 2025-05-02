FROM nvidia/cuda:11.8.0-base-ubuntu20.04
# install python

RUN apt update --quiet \
    && apt install --yes --quiet software-properties-common \
    && apt install --yes --quiet git wget gcc g++ make zlib1g-dev zstd

RUN add-apt-repository ppa:deadsnakes/ppa \
    && DEBIAN_FRONTEND=noninteractive apt install --yes --quiet python3.8 python3-pip python3.8-venv python3.8-dev

RUN python3.8 -m venv /LABind_env
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip3 install --upgrade pip

RUN pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install biopython==1.83 transformers==4.39.3 scikit-learn==1.3.2 pandas==2.0.3 numpy==1.24.3 scipy==1.10.1
RUN pip install lxml==5.2.1 periodictable==1.7.0 accelerate==0.30.1

COPY . /app/LABind
WORKDIR /app/LABind
