# Demos: torchft keeps training through a replica failure

**One sentence:** when one replica dies, the other replicas keep training without
interruption and stay ready, while the failed one is restarted and rejoins by healing
from a healthy peer, with no checkpoint restore.

Both run on this repo's image (the public Neuron DLC + Python 3.12 + Monarch).

| | [Demo: Monarch](docs/demo-monarch.md) (main) | [Demo: torchrun](docs/demo-torchrun.md) |
|---|---|---|
| Manifests | `lighthouse.yaml`, `workers.yaml`, `controller.yaml` | `lighthouse.yaml`, `job.yaml` |
| Training processes started by | the Monarch controller | torchrun in each pod |
| Failure detected by | the controller (crash or hang) | the pod exiting (crash), liveness (hang) |
| Restart done by | Kubernetes (liveness, on the controller's mark) | Kubernetes (the Job) |
| Heal | torchft, from a healthy peer | torchft, from a healthy peer |

Common notes:

- The logged loss is torchft's **sum** over replicas; `METRICS` lines and CloudWatch
  `Loss` divide it by the quorum size.
- Data: C4 by default; `QWEN3_DATASET=c4_test` uses the tiny sample for quick tests.
- Next: a real device fault instead of `kill -9`. The DRA driver taints the device
  (`NoExecute`), Kubernetes evicts the pod, torchft carries on (`fault/device-fault.sh`).
