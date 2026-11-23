"""Cross-node NIXL WRITE into Neuron memory (the transfer torchft/nixl.py uses).
server: python nixl_xnode.py server   -> prints one JSON line (meta, va, dev, n), waits for the writes, verifies
client: python nixl_xnode.py client '<json>'  -> 20 timed WRITEs of N bytes, last one notifies b"done"
"""
import base64, ctypes, json, statistics, sys, time
import torch, torch_neuronx
try:
    from nixl._api import nixl_agent, nixl_agent_config
except ImportError:
    from nixl_cu12._api import nixl_agent, nixl_agent_config

N = 64 << 20


def nrt_va(t):
    cap = torch_neuronx._C._get_nrt_tensor_capsule(t)
    api = ctypes.pythonapi
    api.PyCapsule_GetName.restype, api.PyCapsule_GetName.argtypes = ctypes.c_char_p, [ctypes.py_object]
    api.PyCapsule_GetPointer.restype, api.PyCapsule_GetPointer.argtypes = ctypes.c_void_p, [ctypes.py_object, ctypes.c_char_p]
    ptr = api.PyCapsule_GetPointer(cap, api.PyCapsule_GetName(cap))
    nrt = ctypes.CDLL("libnrt.so.1")
    nrt.nrt_tensor_get_va.restype, nrt.nrt_tensor_get_va.argtypes = ctypes.c_void_p, [ctypes.c_void_p]
    return nrt.nrt_tensor_get_va(ptr) + t.storage_offset() * t.element_size()


dev = torch.device("neuron", torch_neuronx.current_device())
d = dev.index
role = sys.argv[1]
if role == "server":
    buf = torch.zeros(N // 4, dtype=torch.float32, device=dev)
    torch_neuronx.synchronize()
    va = nrt_va(buf)
    a = nixl_agent("xnode-srv", nixl_agent_config(backends=["LIBFABRIC"]))
    a.register_memory(a.get_reg_descs([(va, N, d, "")], "VRAM"))
    print("JSON " + json.dumps({"meta": base64.b64encode(a.get_agent_metadata()).decode(),
                                "va": va, "dev": d, "n": N}), flush=True)
    got, t_end = set(), time.time() + 300
    while b"done" not in got:
        for m in a.get_new_notifs().values():
            got.update(m)
        if time.time() > t_end:
            print("RESULT server: no b'done' notification in 300 s"); sys.exit(1)
        time.sleep(0.001)
    ok = torch.equal(buf.cpu(), torch.arange(N // 4, dtype=torch.float32))
    print(f"RESULT server: data match {ok}", flush=True)
    sys.exit(0 if ok else 1)
else:
    info = json.loads(sys.argv[2])
    src = torch.arange(N // 4, dtype=torch.float32, device=dev)
    torch_neuronx.synchronize()
    sva = nrt_va(src)
    a = nixl_agent("xnode-cli", nixl_agent_config(backends=["LIBFABRIC"]))
    a.register_memory(a.get_reg_descs([(sva, N, d, "")], "VRAM"))
    peer = a.add_remote_agent(base64.b64decode(info["meta"]))
    local = a.get_xfer_descs([(sva, N, d)], "VRAM")
    remote = a.get_xfer_descs([(info["va"], N, info["dev"])], "VRAM")
    times = []
    for i in range(20):
        h = a.initialize_xfer("WRITE", local, remote, peer, b"done" if i == 19 else f"w{i}".encode())
        t0 = time.perf_counter(); st = a.transfer(h)
        t_end = time.time() + 30
        while st not in ("DONE", "ERR"):
            if time.time() > t_end:
                print(f"RESULT client: write {i} timed out"); sys.exit(1)
            st = a.check_xfer_state(h)
        times.append((time.perf_counter() - t0) * 1e3)
        a.release_xfer_handle(h)
        if st != "DONE":
            print(f"RESULT client: write {i}: {st}"); sys.exit(1)
    med = statistics.median(times)
    print(f"RESULT client: 20 x {N >> 20} MiB DONE, median {med:.2f} ms = {N / med / 1e6:.1f} GB/s", flush=True)
