# Stage 1: Build Stage
FROM nvcr.io/nvidia/tensorrt:23.09-py3 AS build

SHELL ["/bin/bash", "-c"] 

# Set timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && apt-get install -q -y tzdata

# Install build dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cmake \
    cppcheck \
    libasio-dev \
    libeigen3-dev \
    libopencv-dev \
    libboost-all-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl -JOLk https://github.com/CrowCpp/Crow/releases/download/v1.2.0/Crow-1.2.0-Linux.deb && \
    dpkg -i Crow-1.2.0-Linux.deb && rm -f Crow-1.2.0-Linux.deb

# Environment setup
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility,video,compute
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64

# Copy source code into the build stage
COPY . ./

# Build botsort library and example
RUN cd /usr/local/lib && cmake /workspace/CMakeLists.txt && make

# Stage 2: Deploy Stage
FROM nvcr.io/nvidia/tensorrt:23.09-py3

SHELL ["/bin/bash", "-c"]

# Set timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && apt-get install -q -y tzdata

# Install runtime dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libopencv-videostab4.5d \
    libboost-filesystem1.74.0 \
    libboost-system1.74.0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only necessary files from the build stage
COPY --from=build /usr/local/lib/botsort/libbotsort.so /usr/local/lib/botsort/
COPY --from=build /usr/local/lib/bin/* /workspace/botsort/
COPY --from=build /workspace/assets/*.onnx /workspace/botsort/
COPY --from=build /workspace/config /workspace/botsort/
COPY --from=build /workspace/examples/data /workspace/botsort/

# Environment setup
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility,video,compute
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:/usr/lib/i386-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
