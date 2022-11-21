
######################################
import problems


cd Attentions-EPDP-DRL
### Start training
python run.py --problem epdp --graph_size 80  --baseline rollout --run_name 'EPDP80_rollout' --model attention --n_encode_layer 3 --n_epochs 100 --attention_type withedge1
python run.py --problem sepdp --graph_size 80  --baseline critic --run_name 'SEPDP80_critic_edge1' --model attention --attention_type withedge1

### Resume training
python run.py --problem sepdp --graph_size 20 --baseline critic --resume 'outputs/sepdp_20/SEPDP20_critic_20210712T124459/epoch-65.pt'  --model GPN --n_epochs 100 
### Evaluation
python eval.py data/epdp/epdp20_test_seed6666.pkl --model 'outputs/epdp_20/EPDP20_rollout_att_20210607T143359/epoch-99.pt' --model_type attention --decode_strategy greedy --eval_batch_size 1 #  --decode_strategy sample --width 1280 --eval_batch_size 1 --val_size 1000
### Generate data
python generate_data.py --name test --problem epdp --graph_sizes 40 --seed 1111 --dataset_size 100

### Or-tools baseline
python epdp_baseline.py ortools10 'data/epdp/epdp80_test_seed1000.pkl' -f --battery_margin 0.2 --problem sepdp


python run.py --problem tsp --graph_size 20  --baseline rollout --run_name 'TSP20_rollout' --model GPN --n_encode_layer 3 --n_epochs 100 



####################################
cd Attentions-OP-DRL
### Start training
python run.py --graph_size 10 --baseline critic --problem sop --run_name 'SOP10_rollout' --n_encode_layers 3
python run.py --graph_size 10 --baseline rollout --problem op --run_name 'OP10_rollout' --n_encode_layers 3
## Evaluation
python eval.py data/sop/sop10_test_seed6666.pkl --model 'outputs/sop_10/SOP10_rollout_20210423T170722/epoch-18.pt' --decode_strategy greedy
python eval.py data/sop/sop10_test_seed6666.pkl --model 'outputs/sop_10/SOP10_rollout_20210423T170722/epoch-91.pt' --decode_strategy sample --width 1280 --eval_batch_size 1
### Generate data
python generate_data.py -f --name test --seed 6666 --is_gaussian False --sigma 1.0


###################################
cd attention-learn-to-route-master
python run.py --graph_size 20 --baseline rollout --problem cvrp --run_name 'CVRP20_rollout'

cd Heterogeneous-Attentions-PDP-DRL-main
python run.py --graph_size 20 --baseline rollout --run_name 'PDP20_rollout'

cd mardam-master
python train.py

cd DS-EPDP-DRL
python train.py

cat jobname.o1402
qsub
myjobsum
cat 1402.mgmt01/1402.mgmt01.OU
qstat
jobsum
myjobsum 
myjobsum all
qdel
showquota

##################################
tensorboard --logdir="C:\Users\s313488\OneDrive - Cranfield University\Documents\Python\Attentions-EPDP-DRL\logs\epdp_20" --host localhost --port 8088

runfile('C:/Users/s313488/OneDrive - Cranfield University/Documents/Python/Attentions-EPDP-DRL/run.py',args="--problem epdp --graph_size 20  --baseline rollout --run_name 'EPDP20_rollout' --model attention --attention_type withedge1 --n_encode_layer 3 --n_epochs 100 ")
runfile('C:/Users/s313488/OneDrive - Cranfield University/Documents/Python/Attentions-EPDP-DRL/eval.py',args="data/epdp/epdp20_test_seed6666.pkl --model 'outputs/epdp_20/EPDP20_rollout_attn_20210607T143359/epoch-99.pt' --model_type attention --attention_type Kool --decode_strategy greedy --eval_batch_size 1 --problem sepdp")

debugfile('C:/Users/s313488/OneDrive - Cranfield University/Documents/Python/Attentions-EPDP-DRL/epdp_baseline.py',args="ortools 'data/epdp/epdp20_test_seed1000.pkl' -f -unable_multiprocessing --problem sepdp")

runfile('C:/Users/s313488/OneDrive - Cranfield University/Documents/Python/Attentions-EPDP-DRL/generate_data.py',args="--name test --problem epdp --graph_sizes 20 --seed 100 --dataset_size 100")


############## COMMANDS FOR GENERALIZATION ANALYSIS ########################
## EPDP20
python eval.py data/epdp/epdp20_test_seed1000.pkl --model 'outputs/epdp_20/EPDP20_rollout_edge1_20220320T161721/epoch-99.pt' --model_type attention --attention_type withedge1 --decode_strategy greedy -f
&&
python eval.py data/epdp/epdp40_test_seed1000.pkl --model 'outputs/epdp_20/EPDP20_rollout_edge1_20220320T161721/epoch-99.pt' --model_type attention --attention_type withedge1 --decode_strategy greedy -f
&&
python eval.py data/epdp/epdp80_test_seed1000.pkl --model 'outputs/epdp_20/EPDP20_rollout_edge1_20220320T161721/epoch-99.pt' --model_type attention --attention_type withedge1 --decode_strategy greedy -f
&&
## EPDP40
python eval.py data/epdp/epdp20_test_seed1000.pkl --model 'outputs/epdp_40/EPDP40_rollout_edge1_20220328T113613/epoch-99.pt' --model_type attention --attention_type withedge1 --decode_strategy greedy -f
&&
python eval.py data/epdp/epdp40_test_seed1000.pkl --model 'outputs/epdp_40/EPDP40_rollout_edge1_20220328T113613/epoch-99.pt' --model_type attention --attention_type withedge1 --decode_strategy greedy -f
&&
python eval.py data/epdp/epdp80_test_seed1000.pkl --model 'outputs/epdp_40/EPDP40_rollout_edge1_20220328T113613/epoch-99.pt' --model_type attention --attention_type withedge1 --decode_strategy greedy -f
&&
## EPDP80
python eval.py data/epdp/epdp20_test_seed1000.pkl --model 'outputs/epdp_80/EPDP80_rollout_edge1_20220921T170737/epoch-99.pt' --model_type attention --attention_type withedge1 --decode_strategy greedy -f
&&
python eval.py data/epdp/epdp40_test_seed1000.pkl --model 'outputs/epdp_80/EPDP80_rollout_edge1_20220921T170737/epoch-99.pt' --model_type attention --attention_type withedge1 --decode_strategy greedy -f
&&
python eval.py data/epdp/epdp80_test_seed1000.pkl --model 'outputs/epdp_80/EPDP80_rollout_edge1_20220921T170737/epoch-99.pt' --model_type attention --attention_type withedge1 --decode_strategy greedy -f

############ PERFORMANCE WITH DIFFERENT WIND MAGNITUDE ##########################
## 
python epdp_baseline.py ortools1 'data/epdp/epdp20_test_seed1000.pkl' -f --battery_margin 0.2 --problem sepdp &&
python epdp_baseline.py ortools10 'data/epdp/epdp40_test_seed1000.pkl' -f --battery_margin 0.2 --problem sepdp &&
python epdp_baseline.py ortools50 'data/epdp/epdp80_test_seed1000.pkl' -f --battery_margin 0.2 --problem sepdp

python epdp_baseline.py ortools1 'data/epdp/epdp20_test_seed1000.pkl' -f --battery_margin 0.3 --problem sepdp &&
python epdp_baseline.py ortools10 'data/epdp/epdp40_test_seed1000.pkl' -f --battery_margin 0.3 --problem sepdp &&
python epdp_baseline.py ortools50 'data/epdp/epdp80_test_seed1000.pkl' -f --battery_margin 0.3 --problem sepdp

python eval.py data/epdp/epdp20_test_seed1000.pkl --model 'outputs/sepdp_20/SEPDP20_critic_att_20210713T143521/epoch-99.pt' --problem sepdp --model_type attention --attention_type Kool --decode_strategy greedy -f &&
python eval.py data/epdp/epdp20_test_seed1000.pkl --model 'outputs/sepdp_20/SEPDP20_critic_edge1_20220321T235654/epoch-99.pt' --problem sepdp --model_type attention --attention_type withedge1 --decode_strategy greedy -f
 
python eval.py data/epdp/epdp40_test_seed1000.pkl --model 'outputs/sepdp_40/SEPDP40_critic_att_20210713T143608/epoch-89.pt' --problem sepdp --model_type attention --attention_type Kool --decode_strategy greedy -f &&
python eval.py data/epdp/epdp40_test_seed1000.pkl --model 'outputs/sepdp_40/SEPDP40_critic_20210712T124750/epoch-99.pt' --problem sepdp --model_type attention --attention_type withedge1 --decode_strategy greedy -f

python eval.py data/epdp/epdp80_test_seed1000.pkl --model 'outputs/sepdp_80/SEPDP80_critic_20210719T182453/epoch-80.pt' --problem sepdp --model_type attention --attention_type Kool --decode_strategy greedy -f &&
python eval.py data/epdp/epdp80_test_seed1000.pkl --model 'outputs/sepdp_80/EPDP80_critic_20220302T231559/epoch-99.pt' --problem sepdp --model_type attention --attention_type withedge1 --decode_strategy greedy -f


### Compare different decoding strategy

python eval.py data/epdp/epdp20_test_seed1000.pkl --model 'outputs/sepdp_20/SEPDP20_critic_edge1_20220321T235654/epoch-99.pt' --problem sepdp --model_type attention --attention_type withedge1 --decode_strategy greedy -f

python eval.py data/epdp/epdp20_test_seed1000.pkl --model 'outputs/sepdp_20/SEPDP20_critic_edge1_20220321T235654/epoch-99.pt' --problem sepdp --model_type attention --attention_type withedge1 --decode_strategy sample --width 1280 -f 
