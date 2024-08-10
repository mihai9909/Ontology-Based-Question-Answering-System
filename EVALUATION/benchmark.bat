scp -i C:\Users\mihai\.ssh\dgx\dgx -P 42509 negomi@193.226.5.207:/mnt/QNAP/negomi/DATASETS/results.sparql .\results.sparql
scp -i C:\Users\mihai\.ssh\dgx\dgx -P 42509 negomi@193.226.5.207:/mnt/QNAP/negomi/DATASETS/results_base_model.sparql .\results_base_model.sparql
scp -i C:\Users\mihai\.ssh\dgx\dgx -P 42509 negomi@193.226.5.207:/mnt/QNAP/negomi/DATASETS/test/data.sparql .\data.sparql

python evaluate.py
