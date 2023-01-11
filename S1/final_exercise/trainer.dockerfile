# Base image
FROM python:3.7-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /Users/haraldskat-rordam/Documents/GitHub/ML_OPS_Harald/S1/final_exercise

COPY requirements.txt requirements.txt
COPY data.py data.py
COPY main.py main.py
COPY model.py model.py
COPY corruptmnist/ corruptmnist/
COPY model/ model/

RUN pip install -r requirements.txt --no-cache-dir

CMD ["main.py"]

ENTRYPOINT ["python", "-u"]