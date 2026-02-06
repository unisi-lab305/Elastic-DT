FROM nvidia/cuda:12.4.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
                build-essential \
                cmake \
                curl \
                g++ \
                wget \
                bzip2 \
                git \
                vim \
                tmux \
                git \
                unzip \
                libosmesa6-dev \
                libgl1-mesa-glx \
                libglfw3 \
                patchelf \
                libglu1-mesa \
                libxext6 \
                libxtst6 \
                libxrender1 \
                libxi6

RUN apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# FIX: Remove defaults channels and use only conda-forge
RUN apt-get update && \
    apt-get install -y mpich && \
    /opt/conda/bin/conda config --remove channels defaults || true && \
    /opt/conda/bin/conda config --add channels conda-forge && \
    /opt/conda/bin/conda config --set channel_priority strict && \
    /opt/conda/bin/conda create -n edt python=3.7 -c conda-forge --override-channels -y && \
    /opt/conda/bin/conda init bash && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN /opt/conda/envs/edt/bin/pip install --upgrade pip

RUN /opt/conda/envs/edt/bin/pip3 install --default-timeout=300 pybullet==3.0.4 \
                    packaging==19.2 \
                    matplotlib==3.1.1 \
                    opencv-python==4.1.2.30 \
                    meshcat==0.0.18 \
                    transformations==2020.1.1 \
                    scikit-image==0.17.2 \
                    gputil==1.4.0 \
                    circle-fit==0.1.3 \
                    ipython \
                    torch \
                    torchvision \
                    torchaudio \
                    'typing_extensions>=4.0.0' \
                    wandb \
                    scikit-learn

RUN apt-get update && apt-get install -y --fix-missing \
	libglib2.0-0 \
	libsm6 \
	libxrender1 \
	libfontconfig1 \
    libcudnn8 \
    libcudnn8-dev

RUN /opt/conda/envs/edt/bin/pip install protobuf==3.20.* \
			sympy

RUN mkdir /root/.mujoco

COPY mujoco210 /root/.mujoco/mujoco210

RUN /opt/conda/envs/edt/bin/pip3 install -U 'mujoco-py<2.2,>=2.1' \
                                             gym

ENV PYTHONPATH=/workspace/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
ENV PIP_DEFAULT_TIMEOUT=1000

# Install TensorFlow separately with longer timeout (large file)
RUN /opt/conda/envs/edt/bin/pip install --default-timeout=1000 \
    tensorflow==2.6.0 \
    keras==2.6.0

# Install other packages
RUN /opt/conda/envs/edt/bin/pip install \
    absl-py==0.12.0 \
    gin-config==0.4.0 \
    matplotlib==3.4.3 \
    mediapy==1.0.3 \
    opencv-python==4.5.3.56 \
    pybullet==3.1.6 \
    scipy==1.7.1 \
    tf-agents==0.11.0rc0 \
    tqdm==4.62.2 \
    gym==0.23.0

RUN /opt/conda/envs/edt/bin/pip install -U numpy

# Fix typing_extensions conflict between tensorflow (requires ~=3.7.4) and wandb (requires >=4.0.0)
# Force upgrade to 4.6+ which is backward compatible with tensorflow
RUN /opt/conda/envs/edt/bin/pip install --upgrade --force-reinstall 'typing_extensions>=4.6.0'

RUN apt-get update && apt-get install -y rustc cargo

RUN /opt/conda/envs/edt/bin/pip install \
    d4rl \
    git+https://github.com/aravindr93/mjrl@master#egg=mjrl \
    timm \
    'cython<3' \
    omegaconf