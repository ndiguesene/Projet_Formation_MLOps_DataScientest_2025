FROM ubuntu:20.04
  
RUN apt-get update && apt-get install python3-pip -y python3-dev

# Set the working directory
WORKDIR /app

# Create the src/data directory structure : -p flag ensures that Docker creates any missing parent directories
RUN mkdir -p src/models/predict
RUN mkdir -p src/features

COPY src/models/predict/requirements.txt ./src/models/predict

# --no-cache-dir ensures that pip doesn't cache the downloaded package files, which will keep your Docker image smaller
RUN pip3 install --no-cache-dir -r ./src/models/predict/requirements.txt

# Copy the Python scripts into the container
COPY ./src/models/predict ./src/models/predict
COPY ./src/features ./src/features

# Copy setup to find packages
COPY ./setup.py ./

# Install the package inside the container
RUN pip3 install .

ENTRYPOINT ["python3", "./src/models/predict/predict.py"]