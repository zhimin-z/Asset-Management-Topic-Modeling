FROM python:3.10

COPY . .

RUN pip3 install -r requirements.txt

CMD ["python", "Code/experiment-2.py"]