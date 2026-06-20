# ── Build stage ───────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ninja-build pkg-config ca-certificates \
    libssl-dev \
    libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc \
    libhiredis-dev \
    libspdlog-dev \
    nlohmann-json3-dev \
    libgtest-dev \
    && rm -rf /var/lib/apt/lists/*

# prometheus-cpp is not in Ubuntu apt; build and install statically.
RUN git clone --depth 1 --branch v1.3.0 \
        https://github.com/jupp0r/prometheus-cpp.git /tmp/prom && \
    cmake -S /tmp/prom -B /tmp/prom/build \
        -DBUILD_SHARED_LIBS=OFF \
        -DENABLE_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build /tmp/prom/build -j$(nproc) && \
    cmake --install /tmp/prom/build && \
    rm -rf /tmp/prom

WORKDIR /app
COPY . .

# Metal is Apple-only; CPU backend used here. Pass -DGGML_CUDA=ON for GPU nodes.
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=OFF \
        -DGGML_CUDA=OFF \
        -DHOMEBREW_PREFIX=/nonexistent \
        -DCMAKE_PREFIX_PATH="/usr;/usr/local" \
    && cmake --build build -j$(nproc) --target inference_server

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime shared libraries only — no compilers or headers.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgrpc++1.51t64 \
    libprotobuf32t64 \
    libhiredis1.0 \
    libssl3 \
    libspdlog1.13 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/build/inference_server /usr/local/bin/inference_server

# prometheus-cpp is statically linked — no runtime dep needed.

EXPOSE 8080
EXPOSE 50051
EXPOSE 9090

ENV MODEL_PATH=/models/model.gguf
ENV REDIS_URL=redis://redis:6379

ENTRYPOINT ["/usr/local/bin/inference_server"]
