FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# System deps + COLMAP (pre-built via apt)
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    colmap \
    libgl1-mesa-glx libglib2.0-0 \
    libboost-filesystem1.74.0 \
    libboost-iostreams1.74.0 \
    libboost-program-options1.74.0 \
    libboost-system1.74.0 \
    libopencv-dev \
    wget curl \
    && rm -rf /var/lib/apt/lists/*

# Install OpenMVS pre-built binaries
RUN wget -q https://github.com/cdcseacave/openMVS/releases/download/v2.3.0/OpenMVS_Ubuntu22_x86_64.tar.gz \
    -O /tmp/openmvs.tar.gz \
    && tar -xzf /tmp/openmvs.tar.gz -C /usr/local/bin/ \
    && rm /tmp/openmvs.tar.gz

# Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8080
CMD ["python3", "app.py"]
