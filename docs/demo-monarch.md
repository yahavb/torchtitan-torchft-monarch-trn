# Demo 2: torchft with a Monarch controller

**One sentence:** when one replica dies, the other replicas keep training without
interruption and stay ready; the Monarch controller decides the failed one must restart,
Kubernetes restarts it, and it rejoins.

Files: `lighthouse.yaml`, `workers.yaml`, `controller.yaml`, `qwen3_pretrain/`.

## Who detects, who decides, who recovers

- **Kubernetes (EKS)** starts every pod, gives the workers their devices, and is the
  only one that recovers: it restarts a pod whose liveness check fails.
- **The Monarch controller** is the brain. The worker pods wait for it to tell them to
  train. It watches each replica and decides when one must restart (a crash, or a hang),
  then marks that pod failed. The pod's liveness check fails on the mark, and Kubernetes
  restarts it. The same controller pattern drives GRPO (`generate`, `forward_backward`).
- **torchft** keeps the other replicas training, and heals the restarted one from a peer.

## 1. The setup (1 min)

- Qwen3-14B from scratch on C4, with TorchTitan on Trainium.
- 4 worker pods = 4 replicas. Each: 4 Neuron devices + 4 EFA, 16 ranks (TP4 x FSDP4).
  Each runs only a Monarch worker and waits.
- 1 controller pod (CPU): attaches to the workers, spawns 16 training processes in each,
  and tells them to train.
- The **lighthouse** (CPU pod) tells the replicas, every step, who is alive.

```
kubectl apply -f lighthouse.yaml
kubectl create configmap qwen3-pretrain-code --from-file=qwen3_pretrain/ --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f workers.yaml
kubectl apply -f controller.yaml
kubectl get po -w            # workers turn 1/1 once step 1 commits (~5 min)
```

## 2. Healthy training (2 min)

- All 4 workers `1/1`.
- `kubectl logs job/qwen3-controller`: "training on 16 processes" for each replica.
- `kubectl logs qwen3-workers-0-<id> | grep METRICS`: loss, tokens/s, quorum size.
- `kubectl logs deploy/qwen3-lighthouse`: `QUORUM CHANGE ... replicas=4`, then steady lines.
- CloudWatch namespace `Qwen3/TorchFT`: the same metrics per replica.

## 3. The fault (1 min)

Only once all 4 workers are `1/1` (see the notes).

```
kubectl logs -f job/qwen3-controller > /tmp/ctl &
kubectl exec qwen3-workers-2-<id> -- sh -c 'kill -9 "$(cat /run/torchft-neuron-m1/train-rank0.pid)"'
kubectl get po -w
```

## 4. Detect, decide, recover (1 min)

Measured 2026-10-20:

| Time | Event |
|---|---|
| 01:08:32 | `kill -9` of rank 0 in worker 2 |
| 01:08:33 | controller detects the crash, marks the pod failed (1 s) |
| ~01:08:37 | liveness fails; Kubernetes stops the pod and creates a new one (4 s) |
| 01:09:08 | controller attaches to the new pod (after it installs Monarch) |
| 01:09:16 | controller tells it to train |
| ~01:13:34 | new pod has compiled and healed: `1/1` |

- Workers 0, 1 and 3 stayed `1/1` throughout, and kept training at 3 replicas.
- The controller's "stop: gave up after 20 s" lines are expected: the old pod was gone.

## 5. The heal (5 min)

Same as demo 1: the new replica compiles, copies weights + optimizer state from one
healthy peer, and the quorum is back to 4.

## Two checks, one signal

Both read the per-step heartbeat:

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

## Measured

- 50 steps, `c4_test` sample (2026-10-19): matches the torchrun run (loss 49.43 at step 1 in
  both, 11.90 vs 11.72 at step 50, 155 tokens/s per rank); identical weights in all 4.
- 5000 steps on C4 (2026-10-21): completed, loss ~3.4 per replica at step 5000,
  155 tokens/s, 6.6 s/step, quorum of 4 the whole run.

## Notes for the presenter

- Inject the fault only after all workers are `1/1`. Before the first step, the others
  wait for all 4 to start (lighthouse `--initial_replicas 4`), up to torchft's 10-minute
  quorum timeout.
- Save logs right after the fault; Kubernetes drops old log lines.
