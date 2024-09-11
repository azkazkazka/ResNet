log_name=log_eval
base_path=path_to_dataset
trained_network=path_to_model.h5
output_filename=results

echo -e "Evaluation process started"

echo -e
echo -e

echo -e Train log: $PWD/${log_name}.txt
echo -e Error log: $PWD/${log_name}_err.txt

echo -e
echo -e

python -u ../main.py --mode eval --scenario LA --trained_network ${trained_network} --base_path ${base_path} --output_filename ${output_filename} > $PWD/log/LA/${log_name}.txt 2> $PWD/log/LA/${log_name}_err.txt

echo -e "Evaluation process finished"
echo -e "Final predictions are in /predictions/LA directory"
