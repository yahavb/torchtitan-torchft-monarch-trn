# torchtitan-torchft-monarch-trn

Fault-tolerant LLM training on AWS Trainium with the PyTorch stack, on EKS:

- **TorchTitan** trains the model (here Qwen3-14B from scratch on C4, TP4 x FSDP4 per replica).
- **torchft** keeps training going when a replica fails: the others continue, and the
  restarted replica heals from a healthy peer and rejoins, with no checkpoint restore.
- **Monarch** is the controller: the worker pods wait for it to tell them to train; it
  watches every replica and decides when one must restart.
- **Kubernetes** owns the pods and devices (Neuron DRA) and does the restarts.

See [DEMO.md](DEMO.md) for the demo and measured timings.

## Image

`public.ecr.aws/e6d8z6l9/torch-neuronx-py312:2.14.0-neuronx-py312-sdk2.33.0-ubuntu26.04`
(pinned by digest in the manifests), built from [images/torch-neuronx-py312](images/torch-neuronx-py312/Dockerfile):
the public Neuron DLC `torch-neuronx:2.14.0-neuronx-py314-sdk2.33.0-ubuntu26.04` plus a
Python 3.12 env (`/opt/py312`) with torch 2.14, torch-neuronx 2.14 rebuilt for 3.12,
neuronx-cc and Monarch. Python 3.12 because Monarch has no Python 3.14 build yet.
`images/torch-neuronx-py312/build-pod.yaml` builds and pushes it with kaniko.

An init container adds the rest at pod start, pinned (`qwen3_pretrain/build_deps.sh`):

| Dependency | Source |
|---|---|
| torchft with the Neuron process group (`torchft/neuron.py`) and `torchft_lighthouse` | `rpsilva-aws/torchft` @ `5dda867` |
| the Neuron torchft trainer (`examples/kubernetes/neuron_m1`) | same repo and commit |
| TorchTitan, and its missing Python dependencies | `pytorch/torchtitan` @ `1ac4653` + the fork's two patches |

The torchft fork is private: the `github-token` secret (key `GITHUB_TOKEN`) must be able to read it.

## Components

| File | What |
|---|---|
| `lighthouse.yaml` | torchft lighthouse: names the live replicas every step (CPU pod); readable summary log |
| `workers.yaml` | 4 worker pods, one torchft replica each: 4 Neuron devices + 4 EFA, 16 ranks; each runs a Monarch worker |
| `controller.yaml` | Monarch controller: tells the workers to train, detects crashes and hangs, marks a failed pod for Kubernetes to restart |
| `job.yaml` | the same 4 replicas without Monarch (one torchrun per pod; hangs caught by liveness) |
| `l-lnc2-trn-efa-rct.yaml` | DRA claim template: 4 Neuron devices + 4 EFA from one device group, LNC=2 |
| `qwen3_pretrain/` | training config and entrypoint, Monarch actors and controller, probes, CloudWatch metrics, dependency build |
| `images/` | the image's Dockerfile and its kaniko build pod |
| `fault/device-fault.sh` | simulate a Neuron device fault (DRA device taint) and repair it |
| `checks/` | one-off image checks |

Health: readiness is per pod ("this replica finished a step recently"), so survivors stay
ready through another replica's failure. With Monarch, the controller decides a replica
failed and marks its pod; the liveness probe fails on the mark and Kubernetes restarts it.

Metrics: each replica logs a compact `METRICS` line and sends loss, grad norm, tokens/s,
MFU, step time, quorum size and steps completed to CloudWatch (`CW_NAMESPACE`).

## Run (Monarch version)

```bash
kubectl create configmap qwen3-pretrain-code --from-file=qwen3_pretrain/ \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f lighthouse.yaml
kubectl apply -f workers.yaml
kubectl apply -f controller.yaml
kubectl get po -w                                    # workers turn 1/1 after step 1
kubectl logs qwen3-workers-0-<id> | grep METRICS      # training metrics
```

Inject a fault once all workers are `1/1`:

```bash
kubectl exec qwen3-workers-2-<id> -- sh -c 'kill -9 "$(cat /run/torchft-neuron-m1/train-rank0.pid)"'
```

Training arguments are the controller's `args` in `controller.yaml` (e.g. `--training.steps`).

## Requirements

- An EKS cluster with Trainium nodes, the Neuron DRA driver (`neuron.aws.com`) and the
  EFA DRA driver (`dra.net`); Kubernetes >= 1.34 with DRA device taints.
- Internet access from the pods: GitHub (dependencies, Rust toolchain) and Hugging Face (C4).
- For CloudWatch: the node role needs `cloudwatch:PutMetricData`.

## Notes

- The model trains on C4 (`allenai/c4`, English), streamed from Hugging Face, with the
  Qwen3 tokenizer. `QWEN3_DATASET=c4_test` switches to TorchTitan's 2000-document sample.
- The logged loss is torchft's sum over the participating replicas; `METRICS` and
  CloudWatch `Loss` divide it by the quorum size.
