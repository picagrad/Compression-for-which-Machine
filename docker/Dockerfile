# syntax=docker/dockerfile:1
# Compression-for-which-Machine: plug-and-play image. Built on the CFM-Model-Zoo image (PyTorch 2.6.0 for CUDA
# 12.4, MMCV 2.1.0 with CUDA ops, MMDetection 3.3.0, MMSegmentation 1.2.2, and CFM-Model-Zoo), as the README's
# install does it: the library first, then this repo. Adds the codec's own dependencies at the versions the
# paper's results were produced with.
#
#   docker build -t cfwm-repro:<version>-cu124 --build-arg CODEC_REF=<tag or commit> - < docker/Dockerfile
#
# The repo is cloned at CODEC_REF into /opt/Compression-for-which-Machine (the working directory) and installed
# editable, so the README's commands run as they are:
#   docker run --gpus all -v /path/to/data:/data -v ~/.cache/mim:/root/.cache/mim \
#       ghcr.io/picagrad/cfwm-repro:<version>-cu124 python -m cfwm_repro.tools.test_rate_guided ...
#
# The image contains NVIDIA software from the CUDA base image, under the NVIDIA Deep Learning Container License
# (its text is in the image at /NGC-DL-CONTAINER-LICENSE) and the CUDA EULA. This software contains source code
# provided by NVIDIA Corporation.

ARG ZOO_IMAGE=ghcr.io/picagrad/cfm-model-zoo:0.1.1-cu124
FROM ${ZOO_IMAGE}

ARG CODEC_REPO=https://github.com/picagrad/Compression-for-which-Machine.git
ARG CODEC_REF=main
# The repository, installed as its README says: the complete pinned list with --no-deps (README, "Library version").
# The CFM-Model-Zoo line is left out because the base image already holds that library, cloned and editable at
# /opt/CFM-Model-Zoo; MMCV's line is a no-op (already built in the base image).
RUN git clone ${CODEC_REPO} /opt/Compression-for-which-Machine && cd /opt/Compression-for-which-Machine && \
    git checkout -B main ${CODEC_REF} && \
    grep -v -E '^cfm-model-zoo' constraints-pinned.txt > /tmp/pins.txt && \
    pip install --no-cache-dir --no-deps -r /tmp/pins.txt -e . && \
    python -c "import numpy, cfwm_repro, cfm_model_zoo, compressai, pytorch_lightning; \
from compressai.entropy_models import EntropyBottleneck; print('cfwm_repro ok, numpy', numpy.__version__)" && \
    rm -rf /root/.cache /tmp/*

ARG VERSION=0.1.1
LABEL org.opencontainers.image.title="cfwm-repro" \
      org.opencontainers.image.description="Reproduces the paper Compression for which Machine: CFM-Model-Zoo on PyTorch 2.6.0 (CUDA 12.4), MMCV 2.1.0, MMDetection 3.3.0, MMSegmentation 1.2.2, plus CompressAI 1.2.8 and PyTorch Lightning 2.6.1" \
      org.opencontainers.image.source="https://github.com/picagrad/Compression-for-which-Machine" \
      org.opencontainers.image.revision="${CODEC_REF}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.licenses="LicenseRef-Compression-for-which-Machine-Academic-Use"
WORKDIR /opt/Compression-for-which-Machine
