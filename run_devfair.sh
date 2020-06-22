module load anaconda3
source activate locomotion-env

for i in {11..20}
do  
	python ./experiment/run_experiment.py seed=$i reward.reward_function.rewards=[$i\*0.1,0.1,0.001] 
done
