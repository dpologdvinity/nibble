# Container image for the nibble CLI: a CPU inference tool that runs GGUF models mounted into the container.
#
#   docker build -t nibble .
#   docker run --rm -v "$PWD/models:/models:ro" nibble chat -m /models/qwen2.5-0.5b-instruct-q4_k_m.gguf
#
# MARCH selects the instruction set the binary targets. The default, x86-64-v3, enables the AVX2/FMA/F16C
# kernels and runs on x86-64-v3 CPUs (Haswell / Excavator or newer; older ones die with SIGILL); use --build-arg MARCH=x86-64 for a portable (but much
# slower, scalar-only) image.

FROM ubuntu:24.04 AS build
ARG MARCH=x86-64-v3
# Revision reported by `nibble --version` (.git is not in the build context): --build-arg GIT_SHA=$(git rev-parse --short HEAD)
ARG GIT_SHA=unknown
ENV NIBBLE_GIT_SHA=${GIT_SHA}
RUN apt-get update \
    && apt-get install -y --no-install-recommends g++ cmake make git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt LICENSE ./
COPY include include
COPY src src
COPY cli cli
COPY tools/gen_git_header.cmake tools/gen_git_header.cmake
RUN cmake -S . -B /build -DCMAKE_BUILD_TYPE=Release -DNIBBLE_BUILD_TESTS=OFF -DNIBBLE_NATIVE=OFF \
        -DCMAKE_CXX_FLAGS="-march=${MARCH}" \
    && cmake --build /build --target nibble_cli -j"$(nproc)" \
    && strip /build/nibble

FROM ubuntu:24.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --no-create-home nibble
COPY --from=build /build/nibble /usr/local/bin/nibble
COPY LICENSE /usr/share/doc/nibble/LICENSE
USER nibble
ENTRYPOINT ["nibble"]
CMD ["--help"]
