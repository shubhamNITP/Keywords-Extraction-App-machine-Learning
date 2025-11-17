#!/bin/bash

echo "Creating model folder..."
mkdir -p models

echo "Downloading model files from GitHub Release..."

curl -L -o models/count_vectorizer.pkl \
https://github.com/shubhamNITP/Keywords-Extraction-App-machine-Learning/releases/download/v1.0.0/count_vectorizer.pkl

curl -L -o models/tfidf_transformer.pkl \
https://github.com/shubhamNITP/Keywords-Extraction-App-machine-Learning/releases/download/v1.0.0/tfidf_transformer.pkl

curl -L -o models/features_names.pkl \
https://github.com/shubhamNITP/Keywords-Extraction-App-machine-Learning/releases/download/v1.0.0/features_names.pkl

echo "Model download complete!"
