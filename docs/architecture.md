# Component diagram: Monarch + torchft on Trainium

```mermaid
flowchart LR
  subgraph CP["Controller pod (CPU)"]
    MC["Monarch controller"]
    LH["torchft Lighthouse"]
  end

  subgraph W0["Worker pod 0"]
    MW["Monarch worker"]
    TR["TorchTitan trainer<br/>16 ranks"]
    FT["torchft Manager<br/>ProcessGroupNeuron"]
    NEU["4 Neuron devices"]
    EFA["4 EFA NICs"]
  end

  W123["Worker pods 1-3<br/>(same as pod 0)"]

  MC -- "Monarch, TCP" --> MW
  MW --> TR
  TR --> NEU
  TR --> FT
  FT -- "quorum, TCP" --> LH
  FT --> EFA
  EFA <== "RDMA: gradients, heal" ==> W123
```

Plain text:

1. Controller pod: Monarch controller and torchft Lighthouse.
2. Four worker pods, all alike. Each has a Monarch worker, the TorchTitan trainer
   (16 ranks), the torchft Manager, 4 Neuron devices and 4 EFA NICs.
3. The controller drives each worker over Monarch (TCP).
4. The torchft Manager asks the Lighthouse for the quorum (TCP).
5. Worker pods exchange gradients, and heal restarted replicas, over EFA RDMA.
