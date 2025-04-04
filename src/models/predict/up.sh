# Run at projet root level (important !)

# build the image
docker build -t predict_app -f src/models/predict/Dockerfile .

# create the container : to the .env file

docker run --name predict_container \
    --env-file .env \
    -v /Users/tiam028713/Documents/Formations/B2B_DataScientest_2025/Projet_Formation_MLOps_DataScientest_2025/models:/app/models \
    -v /Users/tiam028713/Documents/Formations/B2B_DataScientest_2025/Projet_Formation_MLOps_DataScientest_2025/data:/app/data \
    -p 8000:8000 \
    predict_app