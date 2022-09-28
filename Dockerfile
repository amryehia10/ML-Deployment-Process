# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /flask_docker

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 8000
CMD [ "python3","app.py"]