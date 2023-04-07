# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# create a non-root user named appuser, 
# give them the password "appuser" put them in the sudo group
RUN useradd -d /app -m -s /bin/bash appuser && echo "appuser:appuser" | chpasswd && adduser appuser sudo

# Define working directory
WORKDIR /app

# Make the files owned by appuser
RUN chown -R appuser:appuser /app

# Switch to your new user in the docker image
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "Code/experiment_1.py"]
