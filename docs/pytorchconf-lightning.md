# PyTorch Conference CFP draft: lightning talk

Format: Lightning Talk, 1 speaker (Yahav Biran)
Topic area: distributed training / hardware and ecosystem (pick the closest track on the CFP form)

## Title

Connecting the PyTorch Dots on Trainium: torchft, Monarch, and TorchTitan on AWS Neuron

## Abstract

PyTorch now has a full fault-tolerant training stack. Monarch supervises the job from a single controller. torchft keeps healthy replicas training and heals failed ones from a peer, with no checkpoint. TorchTitan runs the model. The community has shown this stack on NVIDIA and AMD GPUs. This talk brings it to AWS Trainium, and shows what the Neuron team built to connect each piece.

We follow one chip fault end to end. Neuron detects the fault and Kubernetes moves the work off the bad chip. Monarch restarts the replica in place. torchft heals it over EFA RDMA through a new Neuron process group. TorchTitan resumes on Neuron with the compiled graph still cached. The healthy replicas never stop, and the restarted replica rejoins in 74 seconds instead of 4 minutes.

## What attendees learn

1. How a new accelerator plgs into torchft and Monarch: a torchft process group, collectives that can cancel transfers to a dead peer, and a cached compiled graph so a restart skips recompiling.
2. That the same PyTorch fault-tolerance code runs unchanged across NVIDIA, AMD, and Trainium.
3. Measured results from a live Trainium cluster, and the lessons we learned the hard way.

## Neuron pieces to name (link public code before submitting)

- torchft Neuron process group (`torchft/neuron.py`)
- torch-neuronx can now cancel collectives to a dead peer
- Monarch on Neuron (Python 3.12 Neuron image)
- TorchTitan Neuron attention adapter
- Neuron DRA driver and Neuron Observer for device health

