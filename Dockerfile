FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update && apt-get install -y curl libopencv-dev

ARG work_dir="/work"

WORKDIR ${work_dir}

ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VERSION="1.2.0"

RUN pip install --upgrade pip

RUN curl -sSSL https://install.python-poetry.org | python3 - && \
    cd /usr/local/bin && \
    ln -s /opt/poetry/bin/poetry && \
    poetry config virtualenvs.create false

COPY ./pyproject.toml* ./poetry.lock* ./

RUN poetry install

ENV PYTHONPATH ${work_dir}:${PYTHONPATH}

CMD [ "jupyter-lab", "--ip", "0.0.0.0", "--allow-root", "--NotebookApp.token=''" ]
