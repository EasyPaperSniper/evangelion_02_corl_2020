module load anaconda3
source activate locomotion-env

for i in {22..22}
do  
	python ./experiment/run_experiment.py seed=$i reward.reward_function.rewards=[0.1,0.1,0.00] 
done
