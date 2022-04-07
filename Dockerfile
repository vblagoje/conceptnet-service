FROM nvidia/cuda:11.2.2-runtime-ubuntu20.04
#set up environment
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

WORKDIR /app


COPY ./requirements.txt /app/requirements.txt

RUN pip3 install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt
RUN python3 -m spacy download en_core_web_sm
COPY . /app/

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]
