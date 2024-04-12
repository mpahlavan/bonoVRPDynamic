#FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
FROM jarvice/app-cuda-9.1:9.1-cudnn7-devel-ubuntu16.04
#FROM juliojj/cuda9-cudnn7-p3-tf1.12

#FROM nvidia/cuda:9.1-cudnn7-runtime-ubuntu16.04

RUN apt-get update && \
	apt-get install -y --no-install-recommends \
		build-essential \
		curl && \
	rm -rf /var/lib/apt/lists/*


RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
	chmod +x miniconda.sh && \
	./miniconda.sh -u -b -p /opt/conda && \
	rm miniconda.sh && \
	/opt/conda/bin/conda install -y pytorch torchvision cudatoolkit=9.0 tqdm matplotlib scipy -c pytorch && \
	/opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

#RUN apt-get install nvidia-384 nvidia-modprobe

RUN curl http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.5.tgz -o LKH-3.0.5.tar.gz && \
	tar zxf LKH-3.0.5.tar.gz && \
	cd LKH-3.0.5 && \
	make && \
	mv LKH /usr/local/bin/ && \
	cd .. && \
	rm -rf LKH-3.0.5

WORKDIR /py_ws/marpdan
RUN touch ../__init__.py
ENV PYTHONPATH /py_ws:$PYTHONPATH
COPY . .
VOLUME /py_ws/marpdan/output

