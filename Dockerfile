FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update

RUN apt -y install python3.9 python3-pip wget git

RUN pip3 install --upgrade pip

RUN pip3 install Flask

RUN mkdir /kogpt2_article

RUN cd /kogpt2_article && git clone https://github.com/SKT-AI/KoGPT2.git

RUN cd /kogpt2_article/KoGPT2 && pip3 install -r requirements.txt

RUN cd /kogpt2_article/KoGPT2 && pip3 install .

COPY . /kogpt2_article

RUN rm /kogpt2_article/KoGPT2_checkpoint/KoGPT2_checkpoint.tar

RUN wget -O /kogpt2_article/KoGPT2_checkpoint/KoGPT2_checkpoint.tar https://github.com/bakjiho/KoGPT2-Article-Generator/raw/master/KoGPT2_checkpoint/KoGPT2_checkpoint.tar

CMD cd /kogpt2_article && python3 /kogpt2_article/run.py
