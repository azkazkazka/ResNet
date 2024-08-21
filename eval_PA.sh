echo -e "Evaluation process started"
echo -e "Check the logs in /log/PA directory"

echo -e
echo -e

echo -e "Train log: ~/speech/RESNET/log/PA/log_evaluation_for_test.txt"
echo -e "Error log: ~/speech/RESNET/log/PA/log_evaluation_for_test_err.txt"

echo -e
echo -e

CUDA_VISIBLE_DEVICES=3 python main.py --mode eval --scenario PA --trained_network ~/speech/RESNET/best_model/PA/best_model_0.001_50_only_prosa.h5  > ./log/PA/log_evaluation_for_test.txt 2> ./log/PA/log_evaluation_for_test_err.txt

echo -e "Evaluation process finished"
echo -e "Predictions are in /log_prediction/PA directory"
echo -e "Models are in /best_model/PA directory"