# Build the commit in config/tensorfold.rev from the repository in config/tensorfold.repo, supplied by scripts/build-native.sh.
ARG CUDA_DEVEL=nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04
ARG CUDA_RUNTIME=nvcr.io/nvidia/cuda:13.0.1-base-ubuntu24.04
FROM ${CUDA_DEVEL} AS build
ARG TENSORFOLD_REV
ARG TENSORFOLD_REPO=https://github.com/ashhart/TensorFold.git
ARG ZIG_VERSION=0.17.0
ARG ZIG_URL=
ARG ZIG_TARGET=aarch64-linux-gnu
ARG ZIG_CPU=baseline+sha2+aes
ARG XGRAMMAR_TAG=v0.2.8
ARG XGRAMMAR_COMMIT=97787376faee5ed8466cfad57c99855e4ce2f6aa
ARG JPEG_TURBO_VERSION=3.1.4.1
ARG JPEG_TURBO_SHA256=ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022
ARG WEBP_VERSION=1.6.0
ARG WEBP_SHA256=e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates xz-utils python3 cmake make \
    && rm -rf /var/lib/apt/lists/*
COPY docker/zig/fetch-source.sh /ctx/recipe/docker/zig/fetch-source.sh
COPY scripts/safe_exec.py scripts/safe_quote.py scripts/safe_output.py scripts/redact.py scripts/safe_shell.sh scripts/watchdog_io.py /ctx/recipe/scripts/
COPY docker/zig/source/ /engine-source/
RUN bash /ctx/recipe/docker/zig/fetch-source.sh "$TENSORFOLD_REV" /ctx/src /engine-source/engine.bundle "$TENSORFOLD_REPO"
RUN set -eu; url="${ZIG_URL:-https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz}"; \
    mkdir -p /opt/zig && curl -fsSL "$url" | tar -xJ -C /opt/zig --strip-components=1; \
    test "$(/opt/zig/zig version)" = "$ZIG_VERSION"
ENV PATH=/opt/zig:/usr/local/cuda/bin:$PATH
WORKDIR /ctx/src
RUN set -eu; test "$(cat .zig-version)" = "$ZIG_VERSION"; \
    nvcc -fatbin -O3 -std=c++17 -gencode=arch=compute_121,code=sm_121 \
        -o /tp_mailbox.fatbin zig/src/fabric/tp/mailbox.cu; \
    zig build dsv41-serve -Dtarget="$ZIG_TARGET" -Dcpu="$ZIG_CPU" \
        -Dnvcc=/usr/local/cuda/bin/nvcc -Ddsv41-nvcc=/usr/local/cuda/bin/nvcc \
        -Dtp-fatbin=/tp_mailbox.fatbin -Dsm=121 -Doptimize=ReleaseSafe -j2 --prefix /out; \
    test -x /out/bin/tensorfold-dsv41 && test -x /out/bin/tf-dsv41
# The engine loads these from lib/ beside bin/ (packaging/RUNTIME.md): libtfgrammar for structured output and required
# tool calls (HTTP 400 without it), over the xgrammar release the engine pins; the pinned JPEG and WebP decoders for
# native image input.
RUN set -eu; mkdir -p /out/lib /out/licenses; \
    git clone -q --branch "$XGRAMMAR_TAG" --recurse-submodules https://github.com/mlc-ai/xgrammar /deps/xgrammar; \
    test "$(git -C /deps/xgrammar rev-parse HEAD)" = "$XGRAMMAR_COMMIT"; \
    zig build grammar -Dxgrammar=/deps/xgrammar -Dtarget="$ZIG_TARGET" -Dcpu="$ZIG_CPU" -j2 --prefix /grammar; \
    cp /grammar/native/lib/libtfgrammar.so /out/lib/; \
    cp /deps/xgrammar/LICENSE /out/licenses/xgrammar-LICENSE; cp /deps/xgrammar/NOTICE /out/licenses/xgrammar-NOTICE; \
    cp /deps/xgrammar/3rdparty/dlpack/LICENSE /out/licenses/dlpack-LICENSE; \
    sed -n '1,/\*\//p' /deps/xgrammar/3rdparty/picojson/picojson.h > /out/licenses/picojson-BSD-2-Clause.txt
RUN set -eu; mkdir -p /codecs && cd /codecs; \
    curl -fsSL -o jpeg.tar.gz "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/$JPEG_TURBO_VERSION/libjpeg-turbo-$JPEG_TURBO_VERSION.tar.gz"; \
    echo "$JPEG_TURBO_SHA256  jpeg.tar.gz" | sha256sum -c -; tar -xzf jpeg.tar.gz; \
    cmake -S "libjpeg-turbo-$JPEG_TURBO_VERSION" -B jpeg-build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/codecs/jpeg-turbo -DCMAKE_INSTALL_LIBDIR=lib -DENABLE_STATIC=OFF; \
    cmake --build jpeg-build -j2 --target install; \
    curl -fsSL -o webp.tar.gz "https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-$WEBP_VERSION.tar.gz"; \
    echo "$WEBP_SHA256  webp.tar.gz" | sha256sum -c -; tar -xzf webp.tar.gz; \
    cmake -S "libwebp-$WEBP_VERSION" -B webp-build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX=/codecs/webp -DCMAKE_INSTALL_LIBDIR=lib -DWEBP_BUILD_ANIM_UTILS=OFF \
        -DWEBP_BUILD_CWEBP=OFF -DWEBP_BUILD_DWEBP=OFF -DWEBP_BUILD_GIF2WEBP=OFF -DWEBP_BUILD_IMG2WEBP=OFF \
        -DWEBP_BUILD_VWEBP=OFF -DWEBP_BUILD_WEBPINFO=OFF -DWEBP_BUILD_WEBPMUX=OFF -DWEBP_BUILD_EXTRAS=OFF; \
    cmake --build webp-build -j2 --target install; \
    cp -P /codecs/jpeg-turbo/lib/libturbojpeg.so.0* /codecs/webp/lib/libwebpdecoder.so.3* /out/lib/; \
    cp "libjpeg-turbo-$JPEG_TURBO_VERSION/LICENSE.md" /out/licenses/libjpeg-turbo-LICENSE.md; \
    cp "libjpeg-turbo-$JPEG_TURBO_VERSION/README.ijg" /out/licenses/libjpeg-turbo-README.ijg; \
    for f in COPYING PATENTS AUTHORS; do cp "libwebp-$WEBP_VERSION/$f" "/out/licenses/libwebp-$f"; done
# The libraries carry the engine's ABI and the pinned versions: every tfg_ entry point tf_grammar.cc defines, the
# TurboJPEG 3 calls, and libwebp's decoder at 1.6.0.
RUN python3 -c 'import ctypes, re, sys; \
grammar = re.findall(r"^(?:void\*|void|int) (tfg_\w+)\(", open("/ctx/src/zig/src/grammar/tf_grammar.cc").read(), re.M); \
assert {"tfg_budget", "tfg_budget_ready"} <= set(grammar), grammar; \
need = {"libtfgrammar.so": grammar, \
        "libturbojpeg.so.0": ["tj3Init", "tj3Destroy", "tj3Set", "tj3Get", "tj3DecompressHeader", "tj3Decompress8", "tj3GetErrorCode"], \
        "libwebpdecoder.so.3": ["WebPGetDecoderVersion", "WebPGetInfo", "WebPDecodeRGBAInto"]}; \
libs = {n: ctypes.CDLL("/out/lib/" + n) for n in need}; \
[getattr(libs[n], s) for n, symbols in need.items() for s in symbols]; \
libs["libwebpdecoder.so.3"].WebPGetDecoderVersion.restype = ctypes.c_int; \
assert libs["libwebpdecoder.so.3"].WebPGetDecoderVersion() == 0x010600; \
assert b"libjpeg-turbo version 3.1.4.1 (build " in open("/out/lib/libturbojpeg.so.0", "rb").read(); \
print("runtime libraries:", {n: len(s) for n, s in need.items()})'

FROM ${CUDA_RUNTIME}
ARG TENSORFOLD_REV
# libnccl2 from the CUDA apt repo the nvidia images ship configured; libibverbs + the mlx5 provider for RoCE; cuBLAS /
# cuBLASLt pinned at 13.6.0.2 (libcublas-13-3, the same repo): the build the vision tower's bit-exact proof ran on, dlopened
# only with TF_DSV41_IMAGES=native. Its directory goes on the loader path (the base image's points at its own CUDA).
RUN apt-get update && apt-get install -y --no-install-recommends libnccl2 libibverbs1 ibverbs-providers rdma-core \
        curl ca-certificates libcublas-13-3=13.6.0.2-1 \
    && apt-mark hold libcublas-13-3 \
    && dirname "$(dpkg -L libcublas-13-3 | grep '/libcublas\.so\.13$')" > /etc/ld.so.conf.d/990_cublas-13-3.conf && ldconfig \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /ctx/src/LICENSE /ctx/src/NOTICE /ctx/src/THIRD_PARTY_NOTICES.md /opt/tensorfold/licenses/
COPY --from=build /ctx/src/LICENSES/ /opt/tensorfold/licenses/LICENSES/
COPY --from=build /out/bin/tensorfold-dsv41 /out/bin/tf-dsv41 /opt/tensorfold/bin/
COPY --from=build /out/lib/ /opt/tensorfold/lib/
COPY --from=build /out/licenses/ /opt/tensorfold/licenses/runtime-deps/
COPY --from=build /tp_mailbox.fatbin /opt/tensorfold/share/tp_mailbox.fatbin
# libproof: every library name the engine can dlopen is covered, each resolves here with its whole NEEDED chain, and
# cuBLAS / cuBLASLt are the pinned 13.6.0.2 files by sha256; any miss or other build fails the image build.
# The grammar and image decoders resolve from /opt/tensorfold/lib, where the engine looks first.
COPY <<'EOF' /opt/tensorfold/share/libproof.sh
#!/bin/bash
set -u
need="libnccl.so.2 libibverbs.so.1 libcublas.so.13 libcublasLt.so.13 libtfgrammar.so libturbojpeg.so.0 libwebpdecoder.so.3"
pins="8062c27c479c249292b48c4bc73af1deb22edee4dfff191fb48a7491ac506410  libcublas.so.13.6.0.2
4198b8726952fbf25a4e78f07178d66509aca4919b7324e935fe3915bff3efc8  libcublasLt.so.13.6.0.2"
# platform/optional_libraries.zig: HIP RCCL and FFmpeg video are other features, not requested by this image.
optional=" librccl.so librccl.so.1 libtfvideo.so "
known=" libc.so.6 libm.so.6 libpthread.so.0 libdl.so.2 libcuda.so.1 libcublas.so libcublasLt.so $need "
fail=0
# Token boundaries keep the debug symbol video_library.soname from reading as library.so.
names=$(grep -aohE '\blib[A-Za-z0-9_]+\.so(\.[0-9]+)*\b' /opt/tensorfold/bin/* | sort -u | tr '\n' ' ')
echo "libproof: engine dlopen names: $names"
for n in $names; do
    case "$known" in *" $n "*) continue ;; esac
    case "$optional" in *" $n "*) echo "libproof: NOT_REQUESTED $n (HIP / video)" ;;
        *) echo "libproof: UNCOVERED dlopen name $n"; fail=1 ;; esac
done
grep -aq 'libjpeg-turbo version 3.1.4.1 (build ' /opt/tensorfold/lib/libturbojpeg.so.0 \
    || { echo "libproof: libturbojpeg is not 3.1.4.1"; fail=1; }
for s in $need; do
    f=/opt/tensorfold/lib/$s
    [ -f "$f" ] || f=$(ldconfig -p | awk -v s="$s" '$1 == s {print $NF; exit}')
    [ -n "$f" ] || { echo "libproof: $s MISSING"; fail=1; continue; }
    r=$(readlink -f "$f"); h=$(sha256sum "$r" | cut -c1-64); u=$(ldd "$r" 2>&1 | grep -c 'not found'); note=""
    [ "$u" = 0 ] || { note=" NEEDED NOT RESOLVED"; fail=1; }
    case $s in libcublas*) if printf '%s\n' "$pins" | grep -qx "$h  ${r##*/}"; then note="$note = pin"; else note="$note != PIN"; fail=1; fi ;; esac
    echo "libproof: $s -> $r ${h:0:12}$note"
done
ls /usr/lib/*-linux-gnu/libibverbs/ 2>/dev/null | grep -q mlx5 || { echo "libproof: mlx5 provider MISSING"; fail=1; }
[ "$fail" = 0 ] && echo "LIBPROOF PASS" || echo "LIBPROOF FAIL"
exit "$fail"
EOF
RUN bash /opt/tensorfold/share/libproof.sh
ENV PATH=/opt/tensorfold/bin:$PATH \
    TF_TP_KERNEL_IMAGE=/opt/tensorfold/share/tp_mailbox.fatbin
LABEL org.tensorfold.revision="${TENSORFOLD_REV}"
LABEL org.opencontainers.image.source="https://github.com/ashhart/TensorFold"
