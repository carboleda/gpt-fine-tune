FROM python as base
RUN pip install langchain openai chromadb tiktoken unstructured unstructured[pdf]

# ==============

FROM base AS run
WORKDIR /usr/app

COPY *.py .
COPY ./data ./data

ENTRYPOINT [ "python", "chatgpt.py" ]
