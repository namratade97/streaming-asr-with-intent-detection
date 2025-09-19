FROM docker.melodis.com/soundhound/icefall:torch1.13.0-cuda11.6-updated-v3


# Copy decoder.py into the Docker image
# COPY decoder.py /workspace/icefall/egs/librispeech/ASR/zipformer/decoder.py
# Ensure executable permissions
# RUN chmod +x /workspace/icefall/egs/librispeech/ASR/zipformer/decoder.py

ENV PYTHONPATH /workspace/icefall:$PYTHONPATH

ENV LD_LIBRARY_PATH /opt/conda/lib/stubs:$LD_LIBRARY_PATH

WORKDIR /workspace/icefall
