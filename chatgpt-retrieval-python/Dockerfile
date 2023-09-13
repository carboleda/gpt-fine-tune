FROM python

WORKDIR /usr/app

COPY *.py .
COPY ./data ./data

RUN pip install langchain openai chromadb tiktoken unstructured unstructured[pdf]

ENTRYPOINT [ "python", "chatgpt.py", "¿Cómo puedo beneficiarme del auxilio de cine?" ]