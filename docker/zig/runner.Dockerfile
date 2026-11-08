# The in-model runner layer over an image scripts/build-native.sh built, supplied by scripts/build-runner-image.sh: the
# engine's circuit frontend (clang-18 / lld-18, packaging/runner in the engine), the runner circuit and the graph bank
# the default profile names (config/prod-zig-flash17.env.example). It adds no engine binary; the build fails unless
# the image's circuit route computes the engine's probe cells.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends clang-18 lld-18 \
    && ln -sf clang-18 /usr/bin/clang && ln -sf ld.lld-18 /usr/bin/ld.lld \
    && rm -rf /var/lib/apt/lists/*
COPY runner/calc.tfpc /opt/tensorfold/share/runner/calc.tfpc
COPY runner/graph-v7-powmod/bank/ /opt/tensorfold/share/runner/graph-v7-powmod/bank/
COPY runner/graph-v7-powmod/bank/Qwenputer-MIT.txt /opt/tensorfold/licenses/runtime-deps/Qwenputer-MIT.txt
RUN /opt/tensorfold/bin/tf-runner-frontend-check --require --circuit /opt/tensorfold/share/runner/calc.tfpc
LABEL org.tensorfold.runner.circuit="calc.tfpc" org.tensorfold.runner.graph="graph-v7-powmod"
