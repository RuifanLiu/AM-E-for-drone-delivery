# Edge-enhanced Attentions

An edge-enhanced attention model to capture the heterogeneous relationships among various nodes (e.g. pickup, delivery, charging) in drone delivery. Training with REINFORCE algorithm while in consideration of stochastic wind conditions.

## Paper 
For more details please see our paper:

Liu, R., Shin, H.S. and Tsourdos, A., 2023. Edge-Enhanced Attentions for Drone Delivery in Presence of Winds and Recharging Stations. Journal of Aerospace Information Systems, 20(4), pp.216-228.
[Publisher Link](https://arc.aiaa.org/doi/10.2514/1.I011171)

## Dependencies
- Python
- NumPy
- SciPy
- PyTorch
- tqdm
- tensorboard_logger
- Matplotlib (optional, only for plotting)

## Quick Start 
For training E-PDP instances with 20 nodes and using the proposed AM-E model:
```bash
python run.py --problem epdp --graph_size 20  --baseline rollout --run_name 'EPDP20_rollout' --model attention --attention_type withedge1
```
## Usage

### Training
For training E-PDP instances with 20 nodes and using the proposed AM-E model:
```bash
python run.py --problem epdp --graph_size 20  --baseline rollout --run_name 'EPDP20_rollout' --model attention --attention_type withedge1
```
#### Resume training
You can resume a run using a pre-trained model by using --resume:
```bash
python run.py --problem epdp --graph_size 20 --baseline critic --resume 'outputs/sepdp_20/epoch-65.pt'  --model GPN --n_epochs 100 
```

### Validation

#### Generating data
To generate test data for all problems:
```bash
python generate_data.py --name test --problem epdp --graph_sizes 20 --seed 6666 --dataset_size 1000
```

#### Evaluation
To evaluate the performance, we run eval.py on test data:
```bash
python eval.py data/epdp/epdp20_test_seed6666.pkl --model 'outputs/epdp_20/epoch-99.pt' --model_type attention --decode_strategy greedy --eval_batch_size 1
````
#### Baselines
Baselines using Or-tools can be run as follows:
```bash
python epdp_baseline.py ortools10 'data/epdp/epdp20_test_seed6666.pkl' -f --battery_margin 0.2 --problem epdp
```






