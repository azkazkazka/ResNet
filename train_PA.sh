echo -e "Training process started"
echo -e "Check the logs in /log/PA directory"

echo -e
echo -e

echo -e "Train log: ~/speech/RESNET/log/PA/log_training_only_prosa.txt"
echo -e "Error log: ~/speech/RESNET/log/PA/log_training_only_prosa_err.txt"

echo -e
echo -e

CUDA_VISIBLE_DEVICES=3 python -u ./main.py --mode train --scenario PA > ./log/PA/log_training_only_prosa.txt 2> ./log/PA/log_training_only_prosa_err.txt

echo -e "Training process finished"
echo -e "Predictions are in ./log_prediction/PA directory"
echo -e "Models are in /best_model/PA directory"