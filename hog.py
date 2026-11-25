import argparse
import ctypes
import sys
import time

MIB = 1 << 20


def load_cuda():
    cuda = ctypes.CDLL("libcuda.so.1")
    check(cuda.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
    return cuda


def check(code, what):
    if code != 0:
        print(f"FAIL {what} cuda_error={code}", flush=True)
        sys.exit(2)


def mem_info(cuda):
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    check(cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)), "cuMemGetInfo")
    return free.value // MIB, total.value // MIB


def try_alloc(cuda, mib):
    ptr = ctypes.c_ulonglong()
    code = cuda.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(mib * MIB))
    return code, ptr


def grab(cuda, target_mib, chunk_mib, held):
    got = 0
    while got < target_mib:
        code, ptr = try_alloc(cuda, chunk_mib)
        if code != 0:
            print(f"ALLOC_REFUSED at {got} MiB cuda_error={code}", flush=True)
            return got
        held.append(ptr)
        got += chunk_mib
    return got


def probe(cuda, probe_mib):
    code, ptr = try_alloc(cuda, probe_mib)
    if code == 0:
        cuda.cuMemFree_v2(ptr)
    return code


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True)
    p.add_argument("--target-mib", type=int, required=True)
    p.add_argument("--chunk-mib", type=int, default=64)
    p.add_argument("--hold-seconds", type=int, default=90)
    p.add_argument("--probe-mib", type=int, default=0)
    p.add_argument("--start-delay", type=int, default=0)
    args = p.parse_args()

    cuda = load_cuda()
    free, total = mem_info(cuda)
    print(f"{args.role} START visible_total={total}MiB visible_free={free}MiB", flush=True)

    time.sleep(args.start_delay)
    held = []
    got = grab(cuda, args.target_mib, args.chunk_mib, held)
    print(f"{args.role} HOLDING {got}/{args.target_mib} MiB", flush=True)

    deadline = time.time() + args.hold_seconds
    while time.time() < deadline:
        free, total = mem_info(cuda)
        line = f"{args.role} t={int(deadline - time.time())}s free={free}MiB"
        if args.probe_mib:
            code = probe(cuda, args.probe_mib)
            line += f" probe_{args.probe_mib}MiB={'ok' if code == 0 else 'FAILED_' + str(code)}"
        print(line, flush=True)
        time.sleep(3)


main()
