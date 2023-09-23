FROM tensorflow/tensorflow:latest-gpu
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.8
RUN mkdir /gendock
WORKDIR /gendock
COPY . /gendock/
RUN apt-get update
RUN apt-get install -y redis-server screen
RUN pip --timeout=1000 install --no-cache-dir --upgrade -r /gendock/requirements.txt
RUN pip install celery[redis]

RUN chmod +x /gendock/server-entrypoint.sh
RUN chmod +x /gendock/worker-entrypoint.sh
RUN chmod +x /gendock/runall.sh

ENTRYPOINT [ "/gendock/runall.sh" ]