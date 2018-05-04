# This script just runs all the smaller cleaning scripts

# Author John Sigmon

DATA_DIR="../../data/kaggle-data/"
SCRIPT_DIR="data-cleaning/"
TMP="tmp"
ZIP_URL="http://nlp.stanford.edu/data/wordvecs/glove.6B.zip"
FLAG=$1

if ! [ -e $DATA_DIR"clean_transcripts.csv" ]
then
    python3 "$SCRIPT_DIR"cleanText.py
    printf "\n"
fi

if ! [ -e $DATA_DIR"merged_data.csv" ]
then
    python3 "$SCRIPT_DIR"joinData.py
    printf "\n"
fi

if [ "$FLAG" = "-g" ]
then
    # Get glove embeddings
    echo Downloading glove embeddings ..
    wget $ZIP_URL -O $TMP
    printf $TMP
    unzip $TMP -d $DATA_DIR 
    rm $TMP
    echo Download finished
fi

