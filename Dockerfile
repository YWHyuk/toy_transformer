FROM nvcr.io/nvidia/pytorch:22.10-py3
RUN python -m spacy download en_core_web_sm && python -m spacy download de_core_news_sm
