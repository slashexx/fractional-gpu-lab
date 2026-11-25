import argparse
import ctypes
import statistics
import time

PTX = b"""
.version 7.0
.target sm_50
.address_size 64
.visible .entry spin(.param .u64 out, .param .u32 iters)
{
  .reg .u64 %rd<3>;
  .reg .u32 %r<6>;
  .reg .f32 %f<3>;
  .reg .pred %p<2>;
  ld.param.u64 %rd1, [out];
  ld.param.u32 %r1, [iters];
  mov.u32 %r2, %tid.x;
  mov.u32 %r3, %ctaid.x;
  mov.u32 %r4, %ntid.x;
  mad.lo.u32 %r5, %r3, %r4, %r2;
  cvt.rn.f32.u32 %f1, %r5;
  mov.f32 %f2, 0f3F800000;
  mov.u32 %r2, 0;
LOOP:
  fma.rn.f32 %f1, %f1, %f2, %f2;
  add.u32 %r2, %r2, 1;
  setp.lt.u32 %p1, %r2, %r1;
  @%p1 bra LOOP;
  mul.wide.u32 %rd2, %r5, 4;
  add.u64 %rd1, %rd1, %rd2;
  st.global.f32 [%rd1], %f1;
  ret;
}
"""

BLOCKS, THREADS = 20, 256


def check(code, what):
    if code != 0:
        raise SystemExit(f"FAIL {what} cuda_error={code}")


def setup():
    cuda = ctypes.CDLL("libcuda.so.1")
    check(cuda.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    check(cuda.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
    ctx = ctypes.c_void_p()
    check(cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev), "cuCtxCreate")
    mod, fn, buf = ctypes.c_void_p(), ctypes.c_void_p(), ctypes.c_ulonglong()
    check(cuda.cuModuleLoadData(ctypes.byref(mod), PTX), "cuModuleLoadData")
    check(cuda.cuModuleGetFunction(ctypes.byref(fn), mod, b"spin"), "cuModuleGetFunction")
    check(cuda.cuMemAlloc_v2(ctypes.byref(buf), ctypes.c_size_t(BLOCKS * THREADS * 4)), "cuMemAlloc")
    return cuda, fn, buf


def launch(cuda, fn, buf, iters):
    out, n = ctypes.c_ulonglong(buf.value), ctypes.c_uint(iters)
    args = (ctypes.c_void_p * 2)(ctypes.addressof(out), ctypes.addressof(n))
    check(cuda.cuLaunchKernel(fn, BLOCKS, 1, 1, THREADS, 1, 1, 0, None, args, None), "cuLaunchKernel")
    check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--role", required=True)
    p.add_argument("--seconds", type=int, default=60)
    p.add_argument("--start-delay", type=int, default=0)
    p.add_argument("--iters", type=int, default=5_000_000)
    args = p.parse_args()

    cuda, fn, buf = setup()
    print(f"{args.role} READY delay={args.start_delay}s run={args.seconds}s", flush=True)
    time.sleep(args.start_delay)

    start = last = time.time()
    window = []
    while time.time() - start < args.seconds:
        t0 = time.perf_counter()
        launch(cuda, fn, buf, args.iters)
        window.append((time.perf_counter() - t0) * 1000)
        if time.time() - last >= 3:
            print(f"{args.role} t={int(time.time() - start):>3}s kernels={len(window):>3} "
                  f"median_ms={statistics.median(window):6.1f}", flush=True)
            window, last = [], time.time()
    print(f"{args.role} DONE", flush=True)


main()
