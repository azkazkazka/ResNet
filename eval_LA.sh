echo -e "Evaluation process started"
echo -e "Check the logs in /log/LA directory"

echo -e
echo -e

echo -e "Train log: ~/speech/RESNET/log/LA/solo_log_evaluation_test_only_prosa_to_cv.txt"
echo -e "Error log: ~/speech/RESNET/log/LA/solo_log_evaluation_test_only_prosa_to_cv_err.txt"

echo -e
echo -e

CUDA_VISIBLE_DEVICES=3 python -u main.py --mode eval --scenario LA --trained_network ~/speech/RESNET/best_model/LA/best_model_0.001_50_only_prosa.h5  > ./log/LA/solo_log_evaluation_test_only_prosa_to_cv.txt 2> ./log/LA/solo_log_evaluation_test_only_prosa_to_cv_err.txt

echo -e "Evaluation process finished"
echo -e "Predictions are in /log_prediction/LA directory"
echo -e "Models are in /best_model/LA directory"