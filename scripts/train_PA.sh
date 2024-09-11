log_name=log_train
base_path=path_to_dataset

echo -e "Training process started"

echo -e
echo -e

echo -e Train log: $PWD/log/LA/${log_name}.txt
echo -e Error log: $PWD/log/LA/${log_name}_err.txt

echo -e
echo -e

python -u ../main.py --mode train --scenario PA --base_path ${base_path} > $PWD/log/PA/${log_name}.txt 2> $PWD/log/PA/${log_name}_err.txt

echo -e "Training process finished"
echo -e "Fold predictions are in ./predictions/PA directory"
echo -e "Models are in /models/PA directory"