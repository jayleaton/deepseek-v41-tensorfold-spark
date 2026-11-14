# The CUDA 13.3 forward-compatibility layer over the runner image (scripts/build-compat-image.sh supplies the context):
# NVIDIA's user-mode driver files (libcuda 610.43.02) as NVIDIA's NGC PyTorch 26.07 container ships them in
# /usr/local/cuda-13.3/compat/lib.real, checked against compat.sha256 and compat.links, in /opt/tensorfold/compat, and
# the entrypoint tf-compat-entry. The entry puts them first on the loader path only while the host kernel module is
# older than 610 and cuInit succeeds through them; otherwise the host driver. TF_CUDA_COMPAT=0 forces the host driver.
# No engine file changes; the engine's revision and tree labels come from the base image.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
COPY compat/ /opt/tensorfold/compat/
COPY tf-compat-entry /opt/tensorfold/bin/tf-compat-entry
RUN test -f /opt/tensorfold/compat/libcuda.so.610.43.02 && test -e /opt/tensorfold/compat/libcuda.so.1 \
    && test -x /opt/tensorfold/bin/tf-compat-entry && sh -n /opt/tensorfold/bin/tf-compat-entry \
    && command -v python3 > /dev/null
LABEL org.tensorfold.cuda-compat="610.43.02 (NVIDIA NGC PyTorch 26.07, cuda-13.3/compat/lib.real)"
ENTRYPOINT ["/opt/tensorfold/bin/tf-compat-entry"]
CMD ["/bin/bash"]
