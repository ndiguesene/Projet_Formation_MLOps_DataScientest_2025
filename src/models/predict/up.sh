# build the image
docker build -t predict_on_train .

# create the container : to the .env file, add the following
cd ../../../
docker run --name predict_container \
    --env-file .env \
    -v /path/to/local/models:/app/models \
    -v /path/to/local/data:/app/data \
    -p 8000:8000 \
    predict_on_train