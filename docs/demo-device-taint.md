# Demo: device fault

1. Break the device of a pod: `fault/break-device.sh qwen3-workers-1-5x956`
2. Watch the pod leave: `kubectl get po -w`
3. Fix the device: `fault/fix-device.sh`

For another app, pass that pod's name in step 1.
