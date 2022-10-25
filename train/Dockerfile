FROM registry.inviz.ai/gpu-base

USER root

WORKDIR /app

COPY . .
ENV PYTHONPATH "${PYTHONPATH}:${WORKDIR}"
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

CMD ["python", "./src/main.py"]
