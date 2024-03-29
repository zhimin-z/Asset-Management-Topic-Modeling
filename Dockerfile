# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# 1024 or higher
EXPOSE 1024

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Define working directory
WORKDIR /app

RUN git config --global --add safe.directory /app

CMD ["python", "main.py"]