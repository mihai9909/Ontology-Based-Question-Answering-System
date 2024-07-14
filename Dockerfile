FROM nvcr.io/nvidia/pytorch:23.09-py3

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y

RUN pip install -r requirements.txt
RUN apt update && apt install -y default-jdk locales
RUN locale-gen en_US.UTF-8

CMD ["./start"]

