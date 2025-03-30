FROM ubuntu:20.04
  
RUN apt-get update && apt-get install python3-pip -y python3-dev

# Set the working directory
WORKDIR /app

# Create the src/data directory structure : -p flag ensures that Docker creates any missing parent directories
RUN mkdir -p src/models/serve

COPY src/models/serve/requirements.txt ./src/models/serve

# --no-cache-dir ensures that pip doesn't cache the downloaded package files, which will keep your Docker image smaller
RUN pip3 install --no-cache-dir -r ./src/models/serve/requirements.txt

# Copy the Python scripts into the container
COPY ./src/models/serve ./src/models/serve

# Copy setup to find packages
COPY ./setup.py ./

# Install the package inside the container
RUN pip3 install .

# Expose the port for FastAPI
EXPOSE 8001

CMD ["uvicorn", "src.models.serve.serve_model_fastapi:app", "--host", "0.0.0.0", "--port", "8001"]