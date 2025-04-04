# Run at projet root level (important !)

# build the image
docker build -t predict_app -f src/models/predict/Dockerfile .

# create the container : to the .env file, add the following

docker run --name predict_container \
    --env-file .env \
    -v models/:/app/models \
    -v data/:/app/data \
    -p 8000:8000 \
    predict_app