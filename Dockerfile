FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update

RUN apt -y install python3.9 python3-pip wget git

RUN pip3 install --upgrade pip

RUN pip3 install Flask

RUN mkdir /kogpt2_article

#https://github.com/SKT-AI/KoGPT2/tree/7fe98b54cf2c12ab6ba7fac1e1aa7c87e93790c4
RUN cd /kogpt2_article && git clone https://github.com/SKT-AI/KoGPT2.git && cd KoGPT2 && git reset --hard 7fe98b54cf2c12ab6ba7fac1e1aa7c87e93790c4

RUN cd /kogpt2_article/KoGPT2 && pip3 install -r requirements.txt

RUN cd /kogpt2_article/KoGPT2 && pip3 install .

COPY . /kogpt2_article

RUN rm /kogpt2_article/KoGPT2_checkpoint/KoGPT2_checkpoint.tar

RUN wget -O /kogpt2_article/KoGPT2_checkpoint/KoGPT2_checkpoint.tar http://download.louissoft.kr/KoGPT2_checkpoint.tar

RUN COPY ./kogpt2_news_wiki_ko_cased_818bfa919d.spiece /kogpt2_article/kogpt2_news_wiki_ko_cased_818bfa919d.spiece
CMD cd /kogpt2_article && python3 /kogpt2_article/run.py
