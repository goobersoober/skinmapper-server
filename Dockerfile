FROM colmap/colmap:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV QT_QPA_PLATFORM=offscreen
ENV DISPLAY=

RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    flask gunicorn \
    open3d pillow scipy numpy \
    opencv-python xatlas

WORKDIR /workspace/app

EXPOSE 8080
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "-t", "1800", "--graceful-timeout", "3600", "app:app"]
