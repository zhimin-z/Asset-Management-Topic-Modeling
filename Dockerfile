# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Add a volume
VOLUME ["/app"]

# Define working directory
WORKDIR /app

RUN git config --global safe.directory /app

# Expose port
EXPOSE  8080

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "Code/experiment_1.py"]
