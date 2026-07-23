# syntax=docker/dockerfile:1
FROM python:3.10-slim

# gym-pybullet-drones' pyproject.toml requires Python ^3.10 (verified against
# https://github.com/utiasDSL/gym-pybullet-drones/blob/main/pyproject.toml).

# System dependencies:
#   git              -- required by pip to install gym-pybullet-drones from
#                        GitHub (it is not published on PyPI -- see
#                        requirements.txt for the pinned commit and why).
#   build-essential  -- in case any dependency needs to compile a C
#                        extension; most (pybullet included) ship prebuilt
#                        wheels, but this avoids a surprise failure on a
#                        platform without one.
#   libgl1, libx11-6 -- defensive: this project only ever runs PyBullet in
#                        headless DIRECT mode (gui=False), which does not
#                        require a display, but gym-pybullet-drones' Logger
#                        imports matplotlib.pyplot, and PyBullet's own
#                        import chain touches OpenGL-adjacent shared
#                        libraries on some platforms even when unused.
#                        Included so a clean rebuild doesn't fail on that.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

# Headless matplotlib backend -- there is no display inside the container,
# and gym-pybullet-drones' Logger.plot() imports matplotlib.pyplot. Setting
# this before matplotlib is ever imported makes it use the Agg (raster,
# file-only) backend instead of trying to find a GUI toolkit.
ENV MPLBACKEND=Agg
# Stream stdout immediately rather than buffering -- run.py's staged output
# is meant to be watched live or tailed from `docker logs`.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first so this layer is cached across rebuilds
# that only touch application code, not requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project.
COPY . .

# Telemetry output directory. sim/sim_runner.py also creates this at
# runtime, but pre-creating it here makes bind-mounting it trivial:
#   docker run -v "$(pwd)/logs:/app/logs" ...
RUN mkdir -p /app/logs

# The container's only job is to run the CLI. `--prompt` calls a locally
# running Ollama server on the HOST, not inside this container -- Ollama is
# never installed or started here (see README.md "Docker + Ollama networking"
# for the exact `docker run` flags needed to reach it, which differ by
# platform). No API key of any kind is required anywhere in this image.
#
# Default CMD (no args) runs the offline demo mission -- no LLM call, no
# Ollama server needed to see the rest of the pipeline work end to end,
# since --mission-file skips Stage 1 entirely.
ENTRYPOINT ["python", "run.py"]
CMD ["--mission-file", "demo/example_missions/wedge_sweep.json"]
