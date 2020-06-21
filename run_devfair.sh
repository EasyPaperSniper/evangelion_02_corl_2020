module load anaconda3
source activate locomotion-env

for i in {1..10}
do  
   python ./experiment/run_experiment.py seed=$i 
done