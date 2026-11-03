# Demo: torchft keeps training through a replica failure

**Live demo runbook (Monarch + torchft, ~10 min):** [docs/demo-live.md](docs/demo-live.md)

**One sentence:** when one replica dies, the other replicas keep training without
interruption and stay ready, while Kubernetes restarts the failed one and it rejoins
by healing from a healthy peer, with no checkpoint restore.

| | [Demo: Monarch](docs/demo-monarch.md) | [Demo: torchrun](docs/demo-torchrun.md) |
|---|---|---|
| Manifests | `lighthouse.yaml`, `workers.yaml`, `controller.yaml` | `lighthouse.yaml`, `job.yaml` |
| Training processes started by | the Monarch controller | torchrun in each pod |
| Failure detected by | the controller (crash or hang) | the pod exiting (crash), liveness (hang) |
| Restart done by | Kubernetes (liveness, on the controller's mark) | Kubernetes (the Job) |
| Heal | torchft, from a healthy peer | torchft, from a healthy peer |

Image: the internal torchft demo image (`torchft-neuron-m1@sha256:0e55895f…`), until a
public torch-neuronx can cancel RDMA transfers to a dead peer. Public torch-neuronx 2.14
(DLC and beta builds checked, 2026-10-23) has no `efa-rdm-device-write-mr-abort-v1`:
survivors' in-flight transfer to the dead replica runs into the 120 s collective timeout
and torchft's Neuron code exits on purpose (`TORCHFT_NEURON_HARD_EXIT`), so all replicas
restart instead of carrying on. `checks/abort-check.yaml` tests an image for it.
On the internal image, Monarch is torchmonarch 0.4.1, installed at pod start.


- The logged loss is torchft's **sum** over replicas; `METRICS` lines and CloudWatch
  `Loss` divide it by the quorum size.
- Data: C4 by default; `QWEN3_DATASET=c4_test` uses the tiny sample for quick tests.
