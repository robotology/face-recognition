# FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Non interactive installation mode
ENV DEBIAN_FRONTEND=noninteractive

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y \
        wget git build-essential ninja-build libglib2.0-0 \
        libsm6 libxrender-dev libxext6 libgl1-mesa-glx
        
# compile python from source in order to get python3.8 - avoid unsupported library problems.
# I could not make mmdeploy work with python3.10 for now

RUN : \
    && apt-get update \ 
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.8-dev \
        python3.8-venv \
        python3.8-distutils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install opencv
RUN apt update && apt install -y libopencv-dev python3-opencv python3-dev

# Install pytorch, necessary for mmpose
RUN pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# Install mmpose dependencies
RUN pip install -U openmim \
    && mim install mmengine \
    && mim install "mmcv>=2.0.0" \
    && mim install "mmdet>=3.0.0"

# Install mmpose
RUN git clone https://github.com/open-mmlab/mmpose.git -bv1.1.0 --depth=1 \
    && cd mmpose \
    && pip install -r requirements.txt \
    && pip install -v -e .

# Install mmdeploy and onnx runtime
RUN pip install mmdeploy==1.0.0 \
    && pip install mmdeploy-runtime-gpu==1.0.0 \
    && pip install onnxruntime-gpu==1.15.1\
    && wget https://github.com/microsoft/onnxruntime/releases/download/v1.15.1/onnxruntime-linux-x64-gpu-1.15.1.tgz \
    && tar -zxvf onnxruntime-linux-x64-gpu-1.15.1.tgz 

ENV ONNXRUNTIME_DIR=/onnxruntime-linux-x64-gpu-1.15.1

# TODO: this requires more investigation as to why we need two different path to libcuda in different part of docker build
ENV LD_LIBRARY_PATH=${ONNXRUNTIME_DIR}/lib:/usr/lib/x86_64-linux-gnu/:/usr/local/cuda-11.7/compat/:$LD_LIBRARY_PATH

# CMake 3.20.0 and add repository if ubuntu < 18.04
RUN wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-linux-x86_64.tar.gz \
    && tar -xzvf cmake-3.20.0-linux-x86_64.tar.gz \
    && ln -sf $(pwd)/cmake-3.20.0-linux-x86_64/bin/* /usr/bin/

ENV CUDA_BIN_PATH=/usr/local/cuda

# MMdeploy sdk prerequisites
RUN git clone https://github.com/openppl-public/ppl.cv.git \
    && cd ppl.cv \
    && git checkout tags/v0.7.0 -b v0.7.0 \
    && ./build.sh cuda

ENV PPLCV_DIR=/ppl.cv

# # Install mmdeploy sdk
RUN git clone -b main https://github.com/open-mmlab/mmdeploy.git --recursive --depth=1 \
    && cd \mmdeploy \
    && mkdir -p build && cd build \
    && cmake .. \
        -DMMDEPLOY_BUILD_SDK=ON \
        -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
        -DMMDEPLOY_BUILD_EXAMPLES=ON \
        -DMMDEPLOY_TARGET_DEVICES=cuda \
        -DMMDEPLOY_TARGET_BACKENDS=ort \
        -Dpplcv_DIR=${PPLCV_DIR}/cuda-build/install/lib/cmake/ppl \
        -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
        -DCUDNN_DIR=${CUDNN_DIR} \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
    && make -j8 && make install

# Install yarp dependencies
RUN apt update && apt install -y cmake g++ wget unzip \
    build-essential git cmake cmake-curses-gui swig \
    libeigen3-dev \
    libace-dev \
    libedit-dev \
    libsqlite3-dev \
    libtinyxml-dev \
    qtbase5-dev qtdeclarative5-dev qtmultimedia5-dev \
    qml-module-qtquick2 qml-module-qtquick-window2 \
    qml-module-qtmultimedia qml-module-qtquick-dialogs \
    qml-module-qtquick-controls qml-module-qt-labs-folderlistmodel \
    qml-module-qt-labs-settings \
    libqcustomplot-dev \
    libgraphviz-dev \
    libjpeg-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav

# Install ffmpeg libraries for video grabber utils
RUN apt update && apt install -y ffmpeg \ 
        libavcodec-dev libavdevice-dev \ 
        libavformat-dev libavutil-dev

RUN cd / && git clone https://github.com/robotology/ycm.git && mkdir robotology && mv ycm/ robotology/ \
    && cd robotology/ycm && mkdir build \
    && cd build && cmake .. && make && make install

# Install yarp
RUN cd /robotology && git clone https://github.com/robotology/yarp.git \
    && cd yarp && mkdir build && cd build \ 
    && cmake -D ENABLE_yarpcar_mjpeg=ON -D ENABLE_yarpmod_ffmpeg_grabber=ON \
        -D YARP_COMPILE_BINDINGS=ON -D CREATE_PYTHON=ON .. \
	&& make -j8 && make install

# Install other yarp libraries
RUN cd /robotology && git clone https://github.com/robotology/icub-main.git \
	&& cd icub-main && mkdir build && cd build && cmake .. && make && make install 
RUN cd /robotology && git clone https://github.com/robotology/icub-contrib-common.git \
	&& cd icub-contrib-common && mkdir build && cd build && cmake .. && make && make install

# These are used for execution and building
ENV PYTHONPATH="/robotology/yarp/build/lib/python3:/yarpRTMPose/src/python"
# ENV CMAKE_PREFIX_PATH="/mmdeploy/build/install/lib/cmake/MMDeploy"

RUN apt install -y vim
# Install the application
ENV MMDEPLOY_DIR=/mmdeploy
ENV MMPOSE_DIR=/mmpose

COPY . /yarpRTMPose

RUN cd /yarpRTMPose && cmake -B build -S . \
   && cd build && make install

RUN /yarpRTMPose/app/scripts/deploy.sh

