# torchtitan-torchft-monarch-trn

Fault-tolerant LLM training on AWS Trainium with the PyTorch stack:

- **TorchTitan** trains the model (here Qwen3-14B from scratch, TP4 x FSDP4 per replica).
- **torchft** keeps training going when a replica fails: the others continue, and the
  replaced replica heals from a healthy peer and rejoins.
- **Monarch** is the controller: the worker pods wait for it to tell them to train; it
  watches every replica and decides when one must restart.
- **Kubernetes (EKS)** owns the pods and devices (Neuron DRA) and does the restarts.

One replica fails, the other replicas keep training and stay ready, Kubernetes restarts
the failed one, and it rejoins. See [DEMO.md](DEMO.md) for the demo and measured timings.

## Components

| File | What |
|---|---|
| `lighthouse.yaml` | torchft lighthouse: names the live replicas every step (CPU pod) |
| `workers.yaml` | 4 worker pods, one torchft replica each: 4 Neuron devices + 4 EFA, 16 ranks; each runs a Monarch worker |
| `controller.yaml` | Monarch controller: tells the workers to train, detects crashes and hangs, marks a failed pod for Kubernetes to restart |
| `job.yaml` | the same 4 replicas without Monarch (one torchrun per pod) |
| `l-lnc2-trn-efa-rct.yaml` | DRA claim template: 4 Neuron devices + 4 EFA from one device group, LNC=2 |
| `qwen3_pretrain/` | training config and entrypoint, Monarch actors and controller, readiness probe, CloudWatch metrics |
| `fault/device-fault.sh` | simulate a Neuron device fault (DRA device taint) and repair it |

Health and recovery: readiness is per pod ("this replica finished a step recently"), so
survivors stay ready through another replica's failure. The controller decides a replica
failed (crash or hang) and marks its pod; the liveness probe fails on that mark and
Kubernetes restarts the pod.

Metrics: each replica logs a compact `METRICS` line and sends loss, grad norm, tokens/s,
MFU, step time, quorum size and steps completed to CloudWatch (`CW_NAMESPACE`).

## Run (Monarch version)

```bash
kubectl apply -f lighthouse.yaml
kubectl create configmap qwen3-pretrain-code --from-file=qwen3_pretrain/ \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f workers.yaml
kubectl apply -f controller.yaml
kubectl get po -w                                   # workers turn 1/1 after step 1 (~5 min)
kubectl logs qwen3-workers-0-<id> | grep METRICS     # training metrics
```

Inject a fault once all workers are `1/1`:

```bash
kubectl exec qwen3-workers-2-<id> -- sh -c 'kill -9 "$(cat /run/torchft-neuron-m1/train-rank0.pid)"'
```

Training arguments are the controller's `args` in `controller.yaml` (e.g. `--training.steps`).

## Requirements

- An EKS cluster with Trainium nodes, the Neuron DRA driver (`neuron.aws.com`) and the
  EFA DRA driver (`dra.net`); Kubernetes >= 1.34 with DRA device taints.
- A training image with torch-neuronx, TorchTitan and torchft with the Neuron process
  group (`torchft-neuron`). Monarch (torchmonarch 0.4.1) is installed at pod start.
- For CloudWatch: the node role needs `cloudwatch:PutMetricData`.

## Notes

- The model trains on C4 (`allenai/c4`, English), streamed from Hugging Face, with the
  Qwen3 tokenizer; the pods need internet access to huggingface.co. `QWEN3_DATASET=c4_test`
  switches to TorchTitan's 2000-document sample in the image for quick tests (the model
  memorizes it within a few hundred steps).
- The logged loss is torchft's sum over the participating replicas; `METRICS` and
  CloudWatch `Loss` divide it by the quorum size.
