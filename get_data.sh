DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
zip_dir="/resources/data/imgs_and_labels.zip"
unzip_dir="/resources/data"
echo $DIR$unzip_dir
echo $DIR$zip_dir
unzip -d $DIR$unzip_dir $DIR$zip_dir
