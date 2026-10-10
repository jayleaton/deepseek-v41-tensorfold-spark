# Build context: src/ is an exported CUDA DeepSeek Zig port; fatbins/ holds optional sm_121 kernels.
# See docs/ZIG-SERVE.md for source availability and required runtime assets.
# must match the nvcc the gated fatbins were built with when fatbins/ is empty (13.x)
ARG CUDA_DEVEL=nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04
ARG CUDA_RUNTIME=nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04

FROM ${CUDA_DEVEL} AS build
ARG ZIG_VERSION=0.17.0
ARG ZIG_URL=
ARG ZIG_TARGET=aarch64-linux-gnu
# the GB10's ARMv8 crypto extensions for the host code: std's SHA-256 (prepared files, PREPARED_VERIFY) runs in software
# on the generic aarch64 baseline (adds instructions only; no float code changes)
ARG ZIG_CPU=baseline+sha2+aes
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates xz-utils \
    && rm -rf /var/lib/apt/lists/*
RUN set -eu; url="${ZIG_URL:-https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz}"; \
    mkdir -p /opt/zig && curl -fsSL "$url" | tar -xJ -C /opt/zig --strip-components=1; \
    /opt/zig/zig version
ENV PATH=/opt/zig:/usr/local/cuda/bin:$PATH
COPY src /ctx/src
COPY fatbins/ /ctx/fatbins/
WORKDIR /ctx/src
RUN set -eu; \
    if ls /ctx/fatbins/dsv41_*.fatbin >/dev/null 2>&1; then \
        echo "using the prebuilt fatbins in /ctx/fatbins"; \
        test -f /ctx/fatbins/tp_mailbox.fatbin; \
        cp /ctx/fatbins/tp_mailbox.fatbin /tp_mailbox.fatbin; \
        gpu="-Dfatbins=/ctx/fatbins -Dtp-fatbin=/tp_mailbox.fatbin"; \
    else \
        echo "no prebuilt fatbins: building with $(nvcc --version | tail -1)"; \
        nvcc -fatbin -O3 -std=c++17 -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_121,code=sm_121 \
            -gencode=arch=compute_90,code=compute_90 -o /tp_mailbox.fatbin zig/src/fabric/tp/mailbox.cu; \
        gpu="-Dnvcc=/usr/local/cuda/bin/nvcc -Dtp-fatbin=/tp_mailbox.fatbin"; \
    fi; \
    zig build -Dtarget="$ZIG_TARGET" -Dcpu="$ZIG_CPU" $gpu -Dsm=121 -Doptimize=ReleaseSafe -j4 --prefix /out install; \
    test -x /out/bin/tensorfold-dsv41 && test -x /out/bin/tf-dsv41-m1

FROM ${CUDA_RUNTIME}
ARG ZIG_COMMIT=unknown
# libnccl2 from the CUDA apt repo the nvidia images ship configured; libibverbs + the mlx5 provider for RoCE
RUN apt-get update && apt-get install -y --no-install-recommends libnccl2 libibverbs1 ibverbs-providers rdma-core \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /out/bin/tensorfold-dsv41 /out/bin/tf-dsv41-m1 /opt/tensorfold/bin/
COPY --from=build /tp_mailbox.fatbin /opt/tensorfold/share/tp_mailbox.fatbin
ENV PATH=/opt/tensorfold/bin:$PATH \
    TF_TP_KERNEL_IMAGE=/opt/tensorfold/share/tp_mailbox.fatbin
LABEL org.tensorfold.zig-commit="${ZIG_COMMIT}"
