FROM python:3.7
COPY . /app
WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
# making directory of app
WORKDIR /app
# copying all files over
COPY . .

CMD gunicorn -w 4 app:app --preload
