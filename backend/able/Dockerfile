FROM python:3.11.9

ENV APPLICATION_NAME=able
ENV VERSION=v1

ENV HOME=/app
ENV PORT=8000

WORKDIR $HOME

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

COPY setup ./setup
COPY src ./src

EXPOSE $PORT

#ENTRYPOINT ["python3", "setup/install.py"]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
