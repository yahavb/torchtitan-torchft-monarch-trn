# Demo 1: torchft on Kubernetes (torchrun)

**One sentence:** when one replica dies, the other replicas keep training without
interruption and stay ready, while Kubernetes replaces the failed one and it rejoins.

Files: `lighthouse.yaml`, `job.yaml`, `qwen3_pretrain/`.
Who does what: **Kubernetes** replaces a failed pod (its Job): the pod exits on a crash,
and fails its liveness check on a hang (no training step for 120 s). **torchft** keeps
the others training and heals the replacement. No controller.

Image: the public Neuron image; an init container builds torchft, the Neuron torchft
trainer and TorchTitan, pinned (see the README).

## 1. The setup (1 min)

- Qwen3-14B from scratch on C4, with TorchTitan on Trainium.
- 4 replicas = 4 pods (one Indexed Job). Each pod: 4 Neuron devices + 4 EFA, 16 ranks
  (TP4 x FSDP4), its own copy of the model, one torchrun.
- The **lighthouse** (one small CPU pod) tells the replicas, every step, who is alive.
- The replicas average gradients over EFA RDMA (torchft's Neuron process group).

```
kubectl create configmap qwen3-pretrain-code --from-file=qwen3_pretrain/ --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f lighthouse.yaml
kubectl apply -f job.yaml
kubectl get po -w            # deps init (~1-2 min), then 1/1 once step 1 commits
```

## 2. Healthy training (2 min)

- All 4 pods `1/1`. Readiness means "this replica finished a step in the last 5 minutes".
- `kubectl logs qwen3-pretrain-0-<id> | grep METRICS`: loss falling, ~155 tokens/s.
- `kubectl logs deploy/qwen3-lighthouse`: one `QUORUM CHANGE ... replicas=4`, then steady lines.
- Same `GRADSYNC` fingerprints in every pod: the replicas are in sync.

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
- Lighthouse: `QUORUM CHANGE ... replicas=3 [0, 2, 3]`.

## 5. The heal (5 min)

- The replacement compiles (~4.5 min), then copies weights + optimizer state from **one
  healthy peer** with the latest step. Its log names the peer
  (`healing is required ... recover_src_replica_rank=N`).
- Quorum back to 4 (`QUORUM CHANGE ... replicas=4`); the replacement turns `1/1`.
- A 1-step catch-up heal may follow a few seconds later (a brief quorum of 3).

## Notes for the presenter

- Use a fresh Job per demo: each failure adds Job backoff before the replacement starts.
- Save logs right after the fault; Kubernetes drops old log lines.
