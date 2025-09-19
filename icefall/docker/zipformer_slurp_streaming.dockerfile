# FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime 
#v3
# FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
#v5


ENV LC_ALL C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive

# Python 3.9
# ARG K2_VERSION="1.24.4.dev20240223+cuda11.6.torch1.13.0"
# ARG KALDIFEAT_VERSION="1.25.4.dev20240223+cuda11.6.torch1.13.0"
# ARG TORCHAUDIO_VERSION="0.13.0+cu116"

ARG K2_VERSION="1.24.4.dev20241030+cuda12.1.torch2.4.0"
ARG KALDIFEAT_VERSION="1.25.4.dev20240725+cuda12.1.torch2.4.0"
ARG TORCHAUDIO_VERSION="2.4.0"

LABEL authors="Fangjun Kuang <csukuangfj@gmail.com>"
LABEL k2_version=${K2_VERSION}
LABEL kaldifeat_version=${KALDIFEAT_VERSION}
LABEL github_repo="https://github.com/k2-fsa/icefall"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ninja-build \
        curl \
        vim \
    	libssl-dev \
        autoconf \
        automake \
        bzip2 \
        ca-certificates \
        ffmpeg \
        g++ \
        gfortran \
        git \
        libtool \
        make \
        patch \
        sox \
        subversion \
        unzip \
        valgrind \
        wget \
        zlib1g-dev \
        && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir \
      torchaudio==${TORCHAUDIO_VERSION} -f https://download.pytorch.org/whl/torch_stable.html \
      k2==${K2_VERSION} -f https://k2-fsa.github.io/k2/cuda.html \
      git+https://github.com/lhotse-speech/lhotse \
      kaldifeat==${KALDIFEAT_VERSION} -f https://csukuangfj.github.io/kaldifeat/cuda.html \
      kaldi_native_io \
      kaldialign \
      kaldifst \
      kaldilm \
      sentencepiece>=0.1.96 \
      tensorboard \
      typeguard \
      dill \
      onnx \
      onnxruntime \
      onnxmltools \
      onnxoptimizer \
      onnxsim \
      multi_quantization \
      numpy \
      pytest \
      graphviz \
      pypinyin

# Copy your modified icefall repository into the container
COPY ./icefall /workspace/icefall

# Set permissions for any new/modified scripts
RUN chmod +x /workspace/icefall/egs/librispeech/ASR/prepare.sh
RUN chmod +x /workspace/icefall/egs/librispeech/ASR/zipformer/decoder.py

WORKDIR /workspace/icefall


ENV PYTHONPATH /workspace/icefall:$PYTHONPATH
ENV LD_LIBRARY_PATH /opt/conda/lib/stubs:$LD_LIBRARY_PATH

# # Install the causal-conv1d package
# # RUN pip install -e ./egs/librispeech/ASR/zipformer/causal-conv1d/
# # RUN pip install --no-cache-dir --force-reinstall -e ./egs/librispeech/ASR/zipformer/causal-conv1d/
# RUN pip install --no-cache-dir --force-reinstall --no-deps -e ./egs/librispeech/ASR/zipformer/causal-conv1d/

# RUN rm -f ./egs/librispeech/ASR/zipformer/causal-conv1d/causal_conv1d/*.so
# ENV CUDA_HOME=/usr/local/cuda

# # Force local compilation against the installed PyTorch
# ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE
# # ENV _GLIBCXX_USE_CXX11_ABI=1
# ENV CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE

# # Manually clean up any previous build attempts
# RUN rm -rf ./egs/librispeech/ASR/zipformer/causal-conv1d/build
# RUN rm -rf ./egs/librispeech/ASR/zipformer/causal-conv1d/causal_conv1d_cuda.cpython-311-x86_64-linux-gnu.so

# RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA version: {torch.version.cuda}, C++ ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')"

# RUN pip install --no-cache-dir --force-reinstall --no-deps -e ./egs/librispeech/ASR/zipformer/causal-conv1d/

# RUN ldd /workspace/icefall/egs/librispeech/ASR/zipformer/causal-conv1d/causal_conv1d_cuda.cpython-311-x86_64-linux-gnu.so




# # Force local compilation against installed PyTorch
# ENV CUDA_HOME=/usr/local/cuda

# # ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE
# # ENV CAUSAL_CONV1D_FORCE_CXX11_ABI=TRUE


# ENV CAUSAL_CONV1D_FORCE_BUILD=TRUE
# # ENV CAUSAL_CONV1D_FORCE_CXX11_ABI=FALSE
# # ENV _GLIBCXX_USE_CXX11_ABI=0

# # Remove old .so files
# RUN rm -f ./egs/librispeech/ASR/zipformer/causal-conv1d/causal_conv1d/*.so
# RUN rm -rf ./egs/librispeech/ASR/zipformer/causal-conv1d/build
# RUN rm -rf ./egs/librispeech/ASR/zipformer/causal-conv1d/causal_conv1d_cuda.cpython-311-x86_64-linux-gnu.so

# RUN rm -rf \
#     ./egs/librispeech/ASR/zipformer/causal-conv1d/build \
#     ./egs/librispeech/ASR/zipformer/causal-conv1d/causal_conv1d/*.so \
#     ./egs/librispeech/ASR/zipformer/causal-conv1d/causal_conv1d_cuda*.so


# # RUN apt-get update && apt-get install -y gcc-11 g++-11 \
# #     && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 11 \
# #     --slave /usr/bin/g++ g++ /usr/bin/g++-11

# # RUN pip install --no-cache-dir --force-reinstall --no-deps -e ./egs/librispeech/ASR/zipformer/causal-conv1d/
# RUN export _GLIBCXX_USE_CXX11_ABI=0 && \
#     pip install --no-cache-dir --force-reinstall --no-deps -e ./egs/librispeech/ASR/zipformer/causal-conv1d/


# RUN rm -rf \
#     ./egs/librispeech/ASR/zipformer/causal-conv1d/build \
#     ./egs/librispeech/ASR/zipformer/causal-conv1d/causal_conv1d/*.so \
#     ./egs/librispeech/ASR/zipformer/causal-conv1d/causal_conv1d_cuda*.so
#     # \
#     # CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0" pip install --no-cache-dir --force-reinstall --no-deps -e ./egs/librispeech/ASR/zipformer/causal-conv1d/

# RUN CXXFLAGS="" pip install --no-cache-dir --force-reinstall --no-deps -e ./egs/librispeech/ASR/zipformer/causal-conv1d/



# RUN python -c "import torch; import causal_conv1d; print('Success! PyTorch ABI:', torch._C._GLIBCXX_USE_CXX11_ABI)"


# # Print PyTorch versions and ABI
# RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA version: {torch.version.cuda}, C++ ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}')"

# # Reinstall causal-conv1d
# # RUN pip install --no-cache-dir --force-reinstall --no-deps -e ./egs/librispeech/ASR/zipformer/causal-conv1d/

# # --- Fix linker path for PyTorch shared libraries ---
# # Print torch install path for debugging
# RUN python -c "import torch, os; print('Torch install dir:', os.path.dirname(torch.__file__))"

# # Add torch/lib to LD_LIBRARY_PATH (adjust Python version if needed)
# ENV LD_LIBRARY_PATH="/opt/conda/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH}"

# # Show the .so files in torch/lib
# RUN ls -lh /opt/conda/lib/python3.11/site-packages/torch/lib

# # Quick import test
# RUN python -c "import torch; import causal_conv1d; print('Success: causal_conv1d loads with torch', torch.__version__)"

