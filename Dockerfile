FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ARG dirname="/work"

RUN apt-get update -y &&\
    apt-get upgrade -y && \
    apt-get install -y vim tmux libgomp1 libgl1-mesa-dev libglib2.0-0 && \
    apt-get install -y python3-dev python3-pip python3-setuptools

RUN mkdir $dirname
WORKDIR $dirname

COPY ./ ./
RUN pip3 install --upgrade pip && \
    pip3 install -r ./requirements.txt
