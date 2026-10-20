# Demo outline: torchft keeps Qwen3 training through a replica failure

**One sentence:** when one replica dies, the other replicas keep training without
interruption and stay ready, while Kubernetes replaces the failed one and it rejoins.

## 1. The setup (1 min)

- Qwen3-14B, pretrained from scratch, on trn3-dev1 (TorchTitan).
- 4 replicas = 4 pods. Each pod has 4 Neuron devices + 4 EFA, 16 ranks, its own copy of the model.
- The **lighthouse** (one small CPU pod) tells the replicas, every step, who is alive.
- The replicas average gradients over EFA RDMA (the demo's Neuron torchft code).

Files: `lighthouse.yaml`, `job.yaml`, `qwen3_pretrain/`.

## 2. Healthy training (2 min)

```
kubectl get po
```
- All 4 pods `1/1`. Readiness means "this replica finished a step in the last 5 minutes".
- Show the loss falling, and identical GRADSYNC fingerprints in every pod (replicas in sync).

## 3. The fault (1 min)

```
kubectl logs -f qwen3-pretrain-0-<id> > /tmp/r0 &
kubectl exec qwen3-pretrain-1-<id> -- sh -c 'kill -9 "$(cat /run/torchft-neuron-m1/torchrun.pid)"'
kubectl get po -w
```
- Pod 1: `Error`. Its Job starts a replacement at once.
- Pods 0, 2, 3: stay `1/1`.

## 4. Training continues (2 min)

Measured 2026-10-19:
- The in-flight step is cancelled and redone (`should_commit=False`); no restart.
- New quorum of 3 (replicas 0, 2, 3) within a second; steps keep advancing.

## 5. The heal (5 min)

- The replacement compiles (~4.5 min), then copies weights + optimizer state from **one
  healthy peer** with the latest step. Its log names the peer
  (`healing is required ... recover_src_replica_rank=N`).
- Quorum back to 4; the replacement turns `1/1`.
- A 1-step catch-up heal may follow a few seconds later (a brief quorum of 3).

## 6. Next: a real device fault

Same result, but triggered by hardware health instead of `kill -9`: the DRA driver
taints the device (`NoExecute`), Kubernetes evicts the pod, torchft carries on.

## Talking point: who detects, who decides, who recovers (Monarch version)

With Monarch (`workers.yaml`, `controller.yaml`), the roles split three ways:

- **Kubernetes (EKS)** starts every pod, gives the workers their devices, and is the
  only one that recovers: it restarts a pod whose liveness check fails.
- **The Monarch controller** is the brain. The worker pods wait for it to tell them to
  train. It watches each replica and decides when one must restart (a crash, or a hang),
  then marks that pod failed. The pod's liveness check fails on the mark, and Kubernetes
  restarts it. The same controller pattern drives GRPO (`generate`, `forward_backward`).
- **torchft** keeps the other replicas training, and heals the restarted one from a peer.

Two checks read the same signal, the per-step heartbeat:

| | Readiness probe | Controller's hang check |
|---|---|---|
| Who runs it | the kubelet, every 5 s | the controller, every 15 s |
| Limit | 300 s without a step | 120 s without a step |
| Result | marks the pod not-ready (reports only) | marks the pod failed, so Kubernetes restarts it (acts) |
| Works if the controller is down | yes | no |

- **Good:** they are independent; readiness keeps reporting even if the controller dies.
- **Watch out:** two limits for one signal. Between 120 s and 300 s the controller has
  decided "failed" while readiness alone would still say ready. The failure mark makes
  readiness not-ready at once, which covers it; one shared limit would be simpler.

Measured 2026-10-19 (50 steps, 4 replicas): the Monarch run matches the torchrun run
(loss 49.43 at step 1 in both, 11.90 vs 11.72 at step 50, 155 tokens/s per rank), and all
4 replicas finished with identical weights.

## Monarch version of the demo

Files: `lighthouse.yaml`, `workers.yaml`, `controller.yaml`, `qwen3_pretrain/`.

```
kubectl apply -f lighthouse.yaml
kubectl create configmap qwen3-pretrain-code --from-file=qwen3_pretrain/ --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f workers.yaml
kubectl apply -f controller.yaml
# wait until all 4 workers are 1/1 (~5 min), then:
kubectl logs -f job/qwen3-controller > /tmp/ctl &
kubectl exec qwen3-workers-2-<id> -- sh -c 'kill -9 "$(cat /run/torchft-neuron-m1/train-rank0.pid)"'
kubectl get po -w
```

Measured 2026-10-20 (4 replicas, fault after all were `1/1`):

| Time | Event |
|---|---|
| 01:08:32 | `kill -9` of rank 0 in worker 2 |
| 01:08:33 | controller detects the crash, marks the pod failed (1 s) |
| ~01:08:37 | liveness fails; Kubernetes stops the pod and creates a new one (4 s) |
| 01:09:08 | controller attaches to the new pod (after it installs Monarch) |
| 01:09:16 | controller tells it to train |
| ~01:13:34 | new pod has compiled and healed: `1/1` |

Workers 0, 1 and 3 stayed `1/1` throughout. The controller's "stop: gave up after 20 s"
lines are expected: the old pod was already gone.

Fault **before** all pods are `1/1` (still compiling): the others wait for all 4 to
start (lighthouse `--initial_replicas 4`), up to torchft's 10-minute quorum timeout.

## Notes for the presenter

- The logged loss is the **sum** over replicas: divide by 3 or 4.
- Save logs right after the fault; Kubernetes drops old log lines.
- Use a fresh Job per demo (each failure adds Job backoff).
