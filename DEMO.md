# Demos: torchft keeps training through a replica failure

**One sentence:** when one replica dies, the other replicas keep training without
interruption and stay ready, while the failed one is restarted and rejoins.

Two versions, same training, same lighthouse:

| | [Demo 1: torchrun](docs/demo-torchrun.md) | [Demo 2: Monarch](docs/demo-monarch.md) |
|---|---|---|
| Manifests | `lighthouse.yaml`, `job.yaml` | `lighthouse.yaml`, `workers.yaml`, `controller.yaml` |
| Training processes started by | torchrun in each pod | the Monarch controller |
| Failure detected by | the pod exiting | the controller (crash or hang) |
| Restart done by | Kubernetes (the Job) | Kubernetes (liveness, on the controller's mark) |
| Heal | torchft, from a healthy peer | torchft, from a healthy peer |

Common to both:

- The logged loss is torchft's **sum** over replicas; `METRICS` lines and CloudWatch
  `Loss` divide it by the quorum size.
- Data: C4 by default; `QWEN3_DATASET=c4_test` uses the tiny in-image sample for quick
  tests (the model memorizes it within a few hundred steps).
- Next: a real device fault instead of `kill -9`. The DRA driver taints the device
  (`NoExecute`), Kubernetes evicts the pod, torchft carries on (`fault/device-fault.sh`).
