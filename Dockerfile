# Container image for the nibble CLI: a CPU inference tool that runs GGUF models mounted into the container.
#
#   docker build -t nibble .
#   docker run --rm -v "$PWD/models:/models:ro" nibble chat -m /models/qwen2.5-0.5b-instruct-q4_k_m.gguf
#
# The binary is built for plain x86-64 and picks its kernels at startup: AVX-VNNI or AVX2 where the CPU has them,
# scalar otherwise, so the image runs on any x86-64 CPU. NIBBLE_KERNELS=scalar|avx2|avx2+vnni forces a tier.

FROM ubuntu:26.04 AS build
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
    && cmake --build /build --target nibble_cli -j"$(nproc)" \
    && strip /build/nibble

FROM ubuntu:26.04
RUN apt-get update \
    && apt-get install -y --no-install-recommends libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --no-create-home nibble
COPY --from=build /build/nibble /usr/local/bin/nibble
COPY LICENSE /usr/share/doc/nibble/LICENSE
USER nibble
ENTRYPOINT ["nibble"]
CMD ["--help"]
