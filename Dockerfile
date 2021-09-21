FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

RUN apt-get -y upgrade & apt-get -y update
RUN apt-get install -y ca-certificates \ 
    curl \ 
    gcc \ 
    build-essential \
    wget \
    libsndfile1 \
    sox \
    ffmpeg

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE 1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED 1

# Add requirements and reinstall torch to required version
RUN pip3 install --upgrade pip

ADD requirements.txt .
RUN pip3 install torchaudio==0.9.0 -f https://download.pytorch.org/whil/torch_stable.html
RUN pip3 install -r requirements.txt

# Set non-root user
# this is to prevent root permission requirements when accessing checkpoints, logs outside of docker container
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

# Set workdir
WORKDIR /youtiao

#docker container starts with bash
RUN ["bash"]