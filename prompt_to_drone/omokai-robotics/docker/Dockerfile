# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile
# Builds the Python pipeline service for the Omokai Robotics demo.
#
# Base:          ubuntu:22.04, pinned by content digest (see FROM line)
# Python:        3.10 (Ubuntu 22.04's system python3 package)
# MAVSDK-Python:  3.10.2 (pinned in requirements.txt)
#
# This image contains ONLY the Python pipeline (planner, validator, executor).
# PX4 SITL + Gazebo Harmonic run in the separate `px4-sitl` service built
# from Dockerfile.px4 — see docker-compose.yml for how they're wired together.
# ─────────────────────────────────────────────────────────────────────────────
FROM ubuntu:22.04@sha256:0e0a0fc6d18feda9db1590da249ac93e8d5abfea8f4c3c0c849ce512b5ef8982

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# ── System dependencies ──────────────────────────────────────────────────────
# python3-dev + build-essential: safety net in case any pinned dependency
# (pydantic, etc.) lacks a prebuilt wheel for this exact platform.
# mavsdk itself ships a manylinux wheel with grpc/protobuf statically
# bundled, so no system protobuf/grpc packages are needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (pinned) ─────────────────────────────────────────────
# Ubuntu 22.04's packaged pip (22.0.2) has a broken PEP 660 editable-install
# check: it reports the setuptools.build_meta backend as "missing the
# build_editable hook" even though setuptools (once resolved to >=64 via
# build isolation) implements it fine. Upgrading pip first resolves it.
RUN pip3 install --no-cache-dir --upgrade pip

# Copy requirements first for layer caching.
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY . .
RUN pip3 install --no-cache-dir -e .

# ── Default command ───────────────────────────────────────────────────────────
# Override with: docker compose run pipeline python3 src/main.py --prompt "..."
CMD ["python3", "src/main.py", "--help"]
