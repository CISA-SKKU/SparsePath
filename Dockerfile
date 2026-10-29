# DANNS — Distributed Approximate Nearest Neighbor Search
# Standalone build: nvidia/cuda base → system deps → conda → gosu
# Multi-user: UID/GID set at container start via entrypoint

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# ============================================================
# Fix GPG key permissions + refresh NVIDIA signing key
# ============================================================
RUN chmod 644 /etc/apt/trusted.gpg.d/*.gpg 2>/dev/null || true && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

# ============================================================
# System packages
# ============================================================
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:git-core/ppa && \
    apt-get update && apt-get install -y \
    # Build tools
    git make cmake g++ ninja-build \
    # DiskANN dependencies (no tcmalloc — conflicts with pybind11)
    libaio-dev libunwind-dev \
    libboost-dev libboost-program-options-dev \
    libmkl-full-dev \
    libomp-dev libomp5 \
    # RDMA libraries
    rdma-core libibverbs-dev librdmacm-dev \
    ibverbs-utils ibverbs-providers perftest infiniband-diags \
    # Utilities
    sudo vim wget curl htop tmux gosu \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# Miniconda
# ============================================================
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_24.4.0-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

ENV PATH=/opt/conda/bin:$PATH

# ============================================================
# Conda environment: danns
# ============================================================
RUN conda create -n danns python=3.10 -y && \
    conda run -n danns conda install -y \
    numpy pandas scipy matplotlib seaborn \
    tqdm pyyaml jupyter ipython \
    scikit-learn pybind11 cmake \
    -c conda-forge && \
    conda clean -tipy

# RAPIDS cuML (GPU K-means), faiss-gpu
RUN conda run -n danns conda install -y \
    -c rapidsai -c conda-forge -c nvidia \
    cuml=24.04 cudf=24.04 rmm=24.04 \
    && conda clean -tipy

RUN conda run -n danns conda install -y \
    -c pytorch -c nvidia \
    faiss-gpu=1.8.0 \
    && conda clean -tipy

RUN conda run -n danns pip install --no-cache-dir \
    rich click typer pytest black isort h5py pyarrow

# Fix conda libtinfo conflict (use system version)
RUN rm -f /opt/conda/envs/danns/lib/libtinfo.so* 2>/dev/null || true

# ============================================================
# Environment
# ============================================================
ENV PATH="/workspace/build/apps:/workspace/build/apps/utils:/opt/conda/envs/danns/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/conda/envs/danns/lib:${LD_LIBRARY_PATH}"
ENV CUDA_HOME="/usr/local/cuda"

WORKDIR /workspace

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
