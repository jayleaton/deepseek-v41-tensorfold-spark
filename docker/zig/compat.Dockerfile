# The CUDA forward-compatibility layer over a serving image, supplied by scripts/build-compat-image.sh: the driver files
# config/cuda-compat.sha256 and config/cuda-compat.links pin, in /opt/tensorfold/compat, and tf-compat-entry ahead of
# the base entrypoint (scripts/cuda_compat.py appends ENTRYPOINT and CMD). It adds no engine binary. The entry probes
# cuInit through python3, which the runtime image installs (docker/zig/Dockerfile).
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY compat/ /opt/tensorfold/compat/
COPY tf-compat-entry /opt/tensorfold/bin/tf-compat-entry
RUN test -f /opt/tensorfold/compat/libcuda.so.610.43.02 && test -e /opt/tensorfold/compat/libcuda.so.1 \
    && test -x /opt/tensorfold/bin/tf-compat-entry && sh -n /opt/tensorfold/bin/tf-compat-entry \
    && command -v python3 > /dev/null
LABEL org.tensorfold.cuda-compat="610.43.02" \
      org.tensorfold.cuda-compat.credit="CUDA 13.3 forward-compat driver as Mia / MiaAI-Lab's dual-Spark image ships it; layer from the lane-template packaging (e679834c7)"
