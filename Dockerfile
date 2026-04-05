FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    colmap \
    libgl1-mesa-glx libglib2.0-0 \
    wget curl git \
    && rm -rf /var/lib/apt/lists/*

# OpenMVS
RUN apt-get update && apt-get install -y \
    libopencv-dev libcgal-dev libboost-all-dev \
    libatlas-base-dev libsuitesparse-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/cdcseacave/openMVS.git /opt/openMVS && \
    mkdir /opt/openMVS/build && cd /opt/openMVS/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DOpenMVS_USE_CUDA=OFF && \
    make -j4 install && \
    rm -rf /opt/openMVS

# Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8080
CMD ["python3", "app.py"]
