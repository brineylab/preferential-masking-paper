#!/bin/bash

# usage
usage() {
    echo "Usage: $0 -t <dataset_name>"
    echo "Available dataset types: pretraining, test_annotations, pair_classification, CoV_classification"
    exit 1
}

# check if no arguments were passed
if [ $# -eq 0 ]; then
    usage
fi

# parse command line arguments
while getopts ":t:" opt; do
    case "${opt}" in
        t)
            DATASET_TYPE=${OPTARG}
            ;;
        \?)
            echo "Invalid option: -${OPTARG}" >&2
            usage
            ;;
        :)
            echo "Option -${OPTARG} requires an argument." >&2
            usage
            ;;
    esac
done

# check if DATASET_TYPE is set
if [ -z "${DATASET_TYPE}" ]; then
    echo "Dataset type is required."
    usage
fi
echo "Dataset type: ${DATASET_TYPE}"

# define Zenodo record information
ZENODO_RECORD_ID="14019655"
BASE_URL="https://zenodo.org/records/${ZENODO_RECORD_ID}"
DATASET_FILENAME=""

# determine the dataset URL/filename based on the type
case $DATASET_TYPE in
    pretraining)
        DATASET_FILENAME="train-eval-test_cdr-mask.tar.gz"
        ;;
    test_annotations)
        DATASET_FILENAME="test-set_annotations.tar.gz"
        ;;
    pair_classification)
        DATASET_FILENAME="pair_classification.tar.gz"
        ;;
    CoV_classification)
        DATASET_FILENAME="CoV_classification.tar.gz"
        ;;
    *)
        echo "Unknown dataset type: ${DATASET_TYPE}"
        usage
        ;;
esac
DATASET_URL="${BASE_URL}/files/${DATASET_FILENAME}?download=1"

# download dataset
echo "Downloading dataset from ${DATASET_URL}..."
curl -o "${DATASET_FILENAME}" -L "${DATASET_URL}"

if [ $? -eq 0 ]; then
    echo "Download completed successfully: ${DATASET_FILENAME}"
else
    echo "Failed to download dataset."
    exit 1
fi

# extract downloaded tar.gz file
echo "Extracting ${DATASET_FILENAME}..."
tar xzvf "${DATASET_FILENAME}"

if [ $? -eq 0 ]; then
    echo "Extraction of ${DATASET_FILENAME} completed successfully."
else
    echo "Failed to extract dataset."
    exit 1
fi

# delete the tar.gz archive
echo "Cleaning up..."
rm "${DATASET_FILENAME}"
echo "Deleted archive: ${DATASET_FILENAME}"

echo "Process for ${DATASET_TYPE} data (${DATASET_FILENAME}) completed successfully!"
