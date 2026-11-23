# KubeCon CFP draft

Track: AI Inference and Infrastructure (backup: Operations + Performance)
Format: Session Presentation, 30 minutes, 2 speakers
Speakers: Yahav Biran + co-speaker

## Title

One Bad Chip, Zero Job Restarts: Self-Healing AI Training with DRA Device Taints and torchft

## Abstract

In large AI jobs, one failing accelerator can stall thousands of healthy ones, and a full restart throws away hours of work. We show how to avoid that using only open source building blocks: Kubernetes DRA and DeviceTaintRule from CNCF, and torchft and Monarch from the PyTorch Foundation.

A DRA driver reports device faults, such as uncorrectable memory errors, link errors, or a chip that stops responding, as DeviceTaintRules. Kubernetes evicts only the affected pod and reschedules it on healthy devices. Monarch detects the failure in about 1 second and picks the cheapest fix: restart in place first, a new pod only when needed. torchft keeps the healthy replicas training and heals the new one by copying weights and optimizer state from a peer, with no checkpoint file.

On our cluster, a failed replica rejoined in 74 seconds instead of 4 minutes, and training never stopped. We demo it live, share what broke along the way, and give you manifests for any DRA-capable accelerator.

## Benefits to the ecosystem

This talk joins two Linux Foundation communities that rarely meet on stage: Kubernetes (CNCF) and torchft and Monarch (PyTorch Foundation). Kubernetes knows which device failed. torchft knows how to keep training without it. We show the handoff between them, step by step, with no vendor-specific APIs.

Attendees leave with:
1. A tiered recovery pattern: health-gate pods before a costly startup, restart in place when the devices are fine, and reschedule only when a DeviceTaintRule says the device is bad.
2. Measured results from a live cluster: detection in about 1 second, rejoin in 74 seconds with in-place restart vs 4 minutes with a new pod, and a 5,000-step run where the healthy replicas kept training through every fault.
3. Lessons learned: the collective library must cancel transfers to a dead peer, or every healthy replica hangs. Two health checks with different limits on the same signal can disagree.
4. Open manifests and a fault-injection script that work with any accelerator whose DRA driver publishes device taints.
