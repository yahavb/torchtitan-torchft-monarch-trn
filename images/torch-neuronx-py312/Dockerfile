# The public Neuron DLC (Python 3.14) plus a Python 3.12 environment, so Monarch
# (wheels for 3.10-3.13 only) can run next to torch 2.14 and torch-neuronx 2.14.
# Recipe from Arash Jamalian; verified with checks/py312-build-check.yaml (2026-10-22):
# torch_neuronx imports on 3.12, its Neuron RDMA backend loads, a Monarch actor runs.
#
# Python 3.12 lives in /opt/py312 (a venv). torch-neuronx is rebuilt for 3.12 in place
# in the DLC's own source tree (/workspace/torch_neuronx): put that directory on
# PYTHONPATH to import it (workers.yaml, job.yaml do).
ARG BASE_IMAGE=public.ecr.aws/neuron/torch-neuronx:2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04
FROM ${BASE_IMAGE}

ARG TORCH_VERSION=2.14.0
# The 3.14 side's Neuron packages, same versions (checks/py312-build-check.yaml).
ARG NEURONX_CC_VERSION=2.28.7631.0+8f7b0cf5
ARG TORCHTITAN_NEURON_VERSION=0.2.2+neuron.1
ARG TORCHMONARCH_VERSION=0.8.0.dev20261022

ENV DEBIAN_FRONTEND=noninteractive
USER root

RUN apt-get update -y \
    && apt-get install -y sudo wget curl gnupg lsb-release software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y python3.12 python3.12-dev python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/py312 \
    && /opt/py312/bin/python -m pip install --no-cache-dir --upgrade pip \
    && /opt/py312/bin/python -m pip install --no-cache-dir \
       "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu

WORKDIR /workspace/torch_neuronx
RUN ./prerequisites.sh \
    && PYTHON_VERSION=3.12 PATH="/opt/py312/bin:${PATH}" ./tools/build --wheel release

RUN /opt/py312/bin/python -m pip install --no-cache-dir \
       --extra-index-url https://pip.repos.neuron.amazonaws.com \
       "neuronx-cc==${NEURONX_CC_VERSION}" \
       "torchtitan-neuron==${TORCHTITAN_NEURON_VERSION}" \
    && /opt/py312/bin/python -m pip install --no-cache-dir --pre \
       "torchmonarch==${TORCHMONARCH_VERSION}"

# Fail the build unless 3.12 has everything this repo needs. No Neuron device during
# a build: keep torch from auto-loading torch-neuronx, and only locate torch_neuronx
# (importing it needs a device; checks/py312-build-check.yaml imports it with one).
RUN export TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    && /opt/py312/bin/python -c "import torch, monarch.actor, neuronxcc; print('torch', torch.__version__)" \
    && PYTHONPATH=/workspace/torch_neuronx /opt/py312/bin/python -c "import importlib.util as u; s = u.find_spec('torch_neuronx'); assert s, 'torch_neuronx not on the 3.12 path'; print('torch_neuronx at', s.origin)" \
    && /opt/py312/bin/python -c "import torch; assert torch.__version__.startswith('${TORCH_VERSION}'), torch.__version__"

WORKDIR /workspace
