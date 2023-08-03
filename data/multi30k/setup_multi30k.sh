git clone --recursive https://github.com/multi30k/dataset.git multi30k-dataset
mv multi30k-dataset/* .
rm -rf multi30k-dataset
mkdir images text
mv data text
mv scripts text
cd text/data/task1/raw
gunzip *.gz
for file in val.*; do mv "$file" "test_2016_val.${file##*.}"; done
cd ../image_splits
mv val.txt test_2016_val.txt
cd ../../../..
mv flickr30k-images.tar.gz images 
mv images_mscoco.task1.tar.gz images
mv test_2017-flickr-images.gz images
cd images
tar -xvzf images_mscoco.task1.tar.gz
mv translated_images test_2017_mscoco
tar -xvzf test_2017-flickr-images.tar.gz
mv task1 test_2017_flickr
tar -xvzf flickr30k-images.tar.gz
mv flickr30k-images train
mkdir test_2016_flickr test_2016_val
mv ../create_test_val_flickr.py .
python create_test_val_flickr.py ../text/data/task1/image_splits/test_2016_val.txt train test_2016_val
python create_test_val_flickr.py ../text/data/task1/image_splits/test_2016_flickr.txt train test_2016_flickr
