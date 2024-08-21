echo -e "Training process started"
echo -e "Check the logs in /log/LA directory"

echo -e
echo -e

echo -e "Train log: ~/speech/RESNET/log/LA/log_kfold_la.txt"
echo -e "Error log: ~/speech/RESNET/log/LA/log_kfold_la_err.txt"

echo -e
echo -e

CUDA_VISIBLE_DEVICES=3 python -u ../main.py --mode train --scenario LA > ./log/LA/log_kfold_la.txt 2> ./log/LA/log_kfold_la_err.txt

echo -e "Training process finished"
echo -e "Predictions are in /log_prediction/LA directory"
echo -e "Models are in /best_model/LA directory"