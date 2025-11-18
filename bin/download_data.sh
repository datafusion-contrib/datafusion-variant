#!/bin/bash

echo "Select the dataset size to download:"
echo "1) 1m (default)"
echo "2) 10m"
echo "3) 100m"
echo "4) 1000m"
read -p "Enter the number corresponding to your choice: " choice

case $choice in
    2)
        # Download 10m dataset: files 0001 to 0010
        mkdir -p ./data/bluesky
        for i in $(seq -f "%04g" 1 10); do
            curl -C - -# --create-dirs -o ./data/bluesky/file_${i}.json.gz "https://clickhouse-public-datasets.s3.amazonaws.com/bluesky/file_${i}.json.gz"
        done
        ;;
    3)
        # Download 100m dataset: files 0001 to 0100
        mkdir -p ./data/bluesky
        for i in $(seq -f "%04g" 1 100); do
            curl -C - -# --create-dirs -o ./data/bluesky/file_${i}.json.gz "https://clickhouse-public-datasets.s3.amazonaws.com/bluesky/file_${i}.json.gz"
        done
        ;;
    4)
        # Download 1000m dataset: files 0001 to 1000
        mkdir -p ./data/bluesky
        for i in $(seq -f "%04g" 1 1000); do
            curl -C - -# --create-dirs -o ./data/bluesky/file_${i}.json.gz "https://clickhouse-public-datasets.s3.amazonaws.com/bluesky/file_${i}.json.gz"
        done
        ;;
    *)
        # Download 1m dataset: single file
        mkdir -p ./data/bluesky
        curl -C - -# --create-dirs -o ./data/bluesky/file_0001.json.gz "https://clickhouse-public-datasets.s3.amazonaws.com/bluesky/file_0001.json.gz"
        ;;
esac
