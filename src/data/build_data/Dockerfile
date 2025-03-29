FROM ubuntu:20.04
  
RUN apt-get update && apt-get install python3-pip -y python3-dev

# Set the working directory
WORKDIR /app

# Create the src/data directory structure : -p flag ensures that Docker creates any missing parent directories
RUN mkdir -p src/data/build_data

COPY src/data/build_data/requirements.txt ./src/data/build_data/

# Copy the Python scripts into the container
COPY src/data/build_data/ ./src/data/build_data

# Copy setup to find packages
COPY ./setup.py ./

# Install the package inside the container
RUN pip3 install .

# --no-cache-dir ensures that pip doesn't cache the downloaded package files, which will keep your Docker image smaller
RUN pip install --no-cache-dir -r ./src/data/build_data/requirements.txt

CMD ["python3", "./src/data/build_data/import_raw_data.py"]