FROM python:3.12

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "main:app"]