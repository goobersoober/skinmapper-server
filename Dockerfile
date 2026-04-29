FROM colmap/colmap:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=
ENV U2NET_HOME=/opt/u2net

RUN apt-get update && apt-get install -y \
    python3-pip git \
    libgl1-mesa-dev \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    flask gunicorn \
    open3d pillow scipy numpy \
    opencv-python xatlas trimesh \
    rembg onnxruntime

# Pre-download u2net foreground segmentation model so first request is fast.
# Without this the first /submit call has to download ~170 MB of model weights.
RUN mkdir -p ${U2NET_HOME} && \
    python3 -c "from rembg import new_session; new_session('u2net')"

WORKDIR /workspace/app

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8888
CMD ["/start.sh"]
