# Live demo runbook: Monarch + torchft on Trainium

The PyTorch Monarch blog's Case Study 2 ("Fault Tolerance in Large Scale Pre Training")
on AWS Trainium: a single Monarch controller, torchft's Lighthouse as a Monarch actor,
fast in-place restart of a failed replica group, and torchft keeping the healthy
replicas stepping. Measured 2026-10-25 (timings below).

**Length:** ~8 min of prep before the audience arrives, then **~10 min live**.

| Part | Time |
|---|---|
| Prep (off stage): start the run, wait for all workers `1/1` | ~8 min |
| 1. The setup | 2 min |
| 2. Healthy training | 1 min |
| 3. The fault, detection, in-place restart | 2 min |
| 4. The heal and rejoin | 2 min |
| 5. Wrap-up | 1-2 min |

## Prep (off stage, ~8 min)

```bash
cd qwen3-pretrain-neuron-deploy
kubectl delete job qwen3-pretrain qwen3-workers qwen3-controller --ignore-not-found
kubectl delete deploy qwen3-lighthouse --ignore-not-found     # the controller serves it
kubectl create configmap qwen3-pretrain-code --from-file=qwen3_pretrain/ \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f workers.yaml
kubectl apply -f controller.yaml
kubectl get po -w | grep --line-buffered qwen3   # wait: all 4 workers 1/1 (~6-8 min)
```

Use a **fresh run** for each demo: each replica gets 2 fast in-place restarts per pod,
and the count does not reset during a run.

Open five terminals, then fill in the pod names (`kubectl get po | grep qwen3-workers`):

| Terminal | Command | Shows |
|---|---|---|
| 1 Pods | `kubectl get po -w \| grep --line-buffered qwen3` | the victim keeps its pod name |
| 2 Controller | `kubectl logs -f job/qwen3-controller \| grep --line-buffered -E "Lighthouse actor\|FAILED\|fast restart\|escalating\|training on\|QUORUM CHANGE"` | detect, decide, quorum |
| 3 Survivor | `kubectl logs -f qwen3-workers-0-<id> \| grep --line-buffered METRICS` | keeps stepping |
| 4 Victim | `kubectl logs -f qwen3-workers-1-<id> \| grep --line-buffered -E "METRICS\|healing is required\|fetching checkpoint"` | heals from a peer |
| 5 Fault | (the kill command, below) | |

## 1. The setup (2 min)

- Qwen3-14B pretraining from scratch on C4 with **TorchTitan**, 4 replicas on Trainium.
  Each replica: one pod with 4 Neuron devices + 4 EFA, 16 ranks (TP4 x FSDP4).
- **Kubernetes** gives each worker pod its devices (Neuron DRA). The workers wait.
- The **Monarch controller** (one CPU pod) is the single control plane: it runs
  **torchft's Lighthouse as a Monarch actor**, spawns each replica group as a Monarch
  process mesh, and tells it to train. Terminal 2 shows `torchft Lighthouse actor serving`.
- **torchft** averages gradients across replicas over EFA RDMA.

## 2. Healthy training (1 min)

- Terminal 3: one `METRICS` line per step, `participants=4`, ~153 tokens/s per rank.
- Terminals 3 and 4: replicas 0 and 1 print the **same loss** every step: in sync.

## 3. The fault (2 min)

Terminal 5:

```bash
kubectl exec qwen3-workers-1-<id> -- sh -c 'kill -9 "$(cat /run/torchft-neuron-m1/train-rank0.pid)"'
```

Measured (2026-10-25, replica 1):

| Time | What | Where |
|---|---|---|
| 18:57:17 | rank 0 killed; quorum drops to 3: `QUORUM CHANGE ... replicas=3 [0, 2, 3]` | T2 |
| 18:57:20 | controller detects it (`Killed(sig=9)`): **fast restart in place (1/2), same pod and devices** | T2 |
| 18:57:34 | new replica group training, **same pod** (never `Error`; goes `0/1`) | T1, T2 |
| meanwhile | survivors keep stepping, `participants=3` | T3 |

Point out in T3: the step in flight at the kill is cancelled and redone (one slow step,
~15 s); then steps continue at 3 replicas.

## 4. The heal and rejoin (2 min)

| Time | What | Where |
|---|---|---|
| 18:58:31 | `healing is required step=0, max_step=556, recover_src_replica_rank=0` | T4 |
| 18:58:31 | `fetching checkpoint from recover_src_replica_rank=0`: weights + optimizer state over RDMA from a healthy peer, **no checkpoint file** | T4 |
| step 557 | the replica is back at the cohort's step, same loss as the others; `QUORUM CHANGE ... replicas=4` | T2, T3, T4 |

**Kill to rejoin: ~74 s.** The restart stays in the same pod, which keeps its devices
and its Neuron compile cache (`/tmp`), so the new processes skip the ~4.5 min compile
that a new pod needs. One slow step (~28 s) while the survivors send their state.

## 5. Wrap-up (1-2 min)

- **Who does what:** Monarch detects and decides; recovery is tiered -- fast in-place
  restart first, and only after 2 failed restarts (or if the pod is gone, e.g. a device
  taint evicts it) it escalates to reallocation: it marks the pod and Kubernetes
  restarts it. torchft keeps the healthy replicas stepping and heals the new one.
- **Without Monarch** (`job.yaml`, torchrun): every failure is a new pod, ~4 min to rejoin (measured 2026-10-19: kill 02:29:51, back in the quorum 02:33:37).

## If something goes wrong

- Survivors stop stepping and all fail after ~2 min: the image's torch-neuronx cannot
  cancel RDMA transfers to a dead peer (public torch-neuronx 2.14). Use the internal
  `torchft-neuron-m1` image (DEMO.md).
- The fault before all workers are `1/1`: the others wait for all 4 to start, up to
  torchft's 10-minute quorum timeout. Inject only after `1/1`.
- A third fault on the same replica escalates to a pod restart (~4-5 min to rejoin).
