# torchtitan-torchft-monarch-trn

Fault-tolerant LLM training on AWS Trainium with the PyTorch stack, on EKS:

- **TorchTitan** trains the model (here Qwen3-14B from scratch on C4, TP4 x FSDP4 per replica).
- **torchft** keeps training going when a replica fails: the others continue, and the
  restarted replica heals from a healthy peer and rejoins, with no checkpoint restore.
- **Monarch** (controller) is parked; see [parked/monarch](parked/monarch/README.md).

See [DEMO.md](DEMO.md) for the demo.

## Image

For now, the internal torchft demo image (`torchft-neuron-m1`, pinned by digest), which
already contains torchft, its Neuron trainer and TorchTitan. Its custom torch-neuronx
cancels RDMA transfers to a dead peer (`efa-rdm-device-write-mr-abort-v1`), which
fault tolerance needs; public torch-neuronx 2.14 does not yet (see DEMO.md).

For the public Neuron DLC, `qwen3_pretrain/build_deps.sh` (run as an init container)
builds the rest, pinned; `images/torch-neuronx-py312` builds a Python 3.12 + Monarch
variant of the DLC. What `build_deps.sh` adds:

| Dependency | Source |
|---|---|
| torchft with the Neuron process group (`torchft/neuron.py`) and `torchft_lighthouse` | `rpsilva-aws/torchft` @ `5dda867` |
| the Neuron torchft trainer (`examples/kubernetes/neuron_m1`) | same repo and commit |
| TorchTitan, and its missing Python dependencies | `pytorch/torchtitan` @ `1ac4653` + the fork's two patches |

The torchft fork is private: the `github-token` secret (key `GITHUB_TOKEN`) must be able
to read it. (Not needed with the internal image.)

## Components

| File | What |
|---|---|
| `lighthouse.yaml` | torchft lighthouse: names the live replicas every step (CPU pod); readable summary log |
| `job.yaml` | 4 replicas (an Indexed Job): each pod has 4 Neuron devices + 4 EFA, 16 ranks, one torchrun; hangs caught by liveness |
| `l-lnc2-trn-efa-rct.yaml` | DRA claim template: 4 Neuron devices + 4 EFA from one device group, LNC=2 |
| `qwen3_pretrain/` | training config and entrypoint, probes, CloudWatch metrics, dependency build |
| `images/` | the Python 3.12 + Monarch image's Dockerfile and its kaniko build pod |
| `parked/monarch/` | the Monarch controller version |
| `fault/device-fault.sh` | simulate a Neuron device fault (DRA device taint) and repair it |
| `checks/` | one-off image checks |

Health, per pod: readiness means "this replica finished a step recently", so survivors
stay ready through another replica's failure; liveness fails when torchrun is gone or
the replica stops stepping for 120 s, and Kubernetes restarts the pod.

Metrics: each replica logs a compact `METRICS` line and sends loss, grad norm, tokens/s,
MFU, step time, quorum size and steps completed to CloudWatch (`CW_NAMESPACE`).

## Run

```bash
kubectl create configmap qwen3-pretrain-code --from-file=qwen3_pretrain/ \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f lighthouse.yaml
kubectl apply -f job.yaml
kubectl get po -w                                   # pods turn 1/1 after step 1
kubectl logs qwen3-pretrain-0-<id> | grep METRICS    # training metrics
```

Inject a fault once all pods are `1/1`:

```bash
kubectl exec qwen3-pretrain-1-<id> -- sh -c 'kill -9 "$(cat /run/torchft-neuron-m1/torchrun.pid)"'
```

Training arguments are `args` in `job.yaml` (e.g. `--training.steps`).

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
