mkdir data result checkpoint
cd data
mkdir train dev raw processed
kaggle datasets download takuok/glove840b300dtxt
unzip glove840b300dtxt.zip
rm glove840b300dtxt.zip
cd train
wget https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip
unzip MINDlarge_train.zip
rm MINDlarge_train.zip
cd ../dev
wget https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip
unzip MINDlarge_dev.zip
head -37647 behaviors.tsv > behaviors.small.tsv
rm MINDlarge_dev.zip
