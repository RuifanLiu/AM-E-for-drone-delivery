#!/usr/bin/env python

import os
import json
import numpy as np

import torch
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from nets.critic_network import CriticNetwork
from utils import torch_load_cpu, load_problem, eval_apriori_routes, count_parameters
from options import get_options
from train import train_epoch, validate, get_inner_model
from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from nets.attention_model import AttentionModel
from nets.attention_model_rnn import AttentionModel_RNN
from nets.pointer_network import PointerNetwork, CriticNetworkLSTM

from baselines._ortools import ort_solve

# from problems import EPDP

def run(opts):

    # Pretty print the run args
    # pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}_{}".format(opts.problem, opts.graph_size), opts.run_name))

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")
    # if opts.use_cuda:
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Figure out what's the problem
    problem = load_problem(opts.problem)
    
    
    # Start the actual training loop
    val_dataset = problem.make_dataset(
        size=opts.graph_size, num_samples=opts.val_size, filename=opts.val_dataset, distribution=opts.data_distribution)

    # solve validate dataset using external baseline methods:
    if opts.ortools_enabled:
        ref_routes, exp_costs = ort_solve(val_dataset,mp=True)
        print("Expect cost on validate dataset (MP) {:5.2f} +- {:5.2f}".format(np.mean(exp_costs), np.std(exp_costs)))
    else:
        ref_routes = None
    if ref_routes is not None:
        ref_costs = eval_apriori_routes(problem, val_dataset, ref_routes, 1 if opts.problem[0] == 's' else 1)
        print("Reference cost on validate dataset {:5.2f} +- {:5.2f}".format(ref_costs.mean(), ref_costs.std()))
        
    if opts.ortools_enabled:
        ref_routes, exp_costs = ort_solve(val_dataset,mp=False)
        print("Expect cost on validate dataset (No-MP) {:5.2f} +- {:5.2f}".format(np.mean(exp_costs), np.std(exp_costs)))
    else:
        ref_routes = None
    if ref_routes is not None:
        ref_costs = eval_apriori_routes(problem, val_dataset, ref_routes, 1 if opts.problem[0] == 's' else 1)
        print("Reference cost on validate dataset {:5.2f} +- {:5.2f}".format(ref_costs.mean(), ref_costs.std()))

    # Load data from load_path
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] Loading data from {}'.format(load_path))
        load_data = torch_load_cpu(load_path)

    # Initialize model
    model_class = {
        'attention': AttentionModel,
        'pointer': PointerNetwork,
        'attention_rnn': AttentionModel_RNN,
    }.get(opts.model, None)
    assert model_class is not None, "Unknown model: {}".format(model_class)
    model = model_class(
        opts.embedding_dim,
        opts.hidden_dim,
        problem,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
        checkpoint_encoder=opts.checkpoint_encoder,
        shrink_size=opts.shrink_size,
        attention_type=opts.attention_type
    ).to(opts.device)
    # Count the number of trainable parameters in model
    count_parameters(model)

    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})

    # Initialize baseline
    if opts.baseline == 'exponential':
        baseline = ExponentialBaseline(opts.exp_beta)
    elif opts.baseline == 'critic' or opts.baseline == 'critic_lstm':
        if problem.NAME == 'tsp':
            baseline = CriticBaseline(
                (
                    CriticNetworkLSTM(
                        2,
                        opts.embedding_dim,
                        opts.hidden_dim,
                        opts.n_encode_layers,
                        
                        opts.tanh_clipping
                    )
                    if opts.baseline == 'critic_lstm'
                    else
                    CriticNetwork(
                        2,
                        opts.embedding_dim,
                        opts.hidden_dim,
                        opts.n_encode_layers,
                        opts.normalization
                    )
                ).to(opts.device)
            )
        elif problem.NAME == 'op' or problem.NAME == 'sop':
            baseline = CriticBaseline(
                (
                    CriticNetwork(
                        opts.embedding_dim,
                        opts.hidden_dim,
                        opts.n_encode_layers,
                        opts.normalization
                    )
                ).to(opts.device)
            )
        elif problem.NAME == 'epdp' or problem.NAME == 'sepdp':
            baseline = CriticBaseline(
                (
                    CriticNetwork(
                        opts.embedding_dim,
                        opts.hidden_dim,
                        opts.n_encode_layers,
                        opts.normalization,
                        opts.attention_type
                    )
                ).to(opts.device)
            )
    elif opts.baseline == 'rollout':
        baseline = RolloutBaseline(model, problem, opts)
    else:
        assert opts.baseline is None, "Unknown baseline: {}".format(opts.baseline)
        baseline = NoBaseline()

    if opts.bl_warmup_epochs > 0:
        baseline = WarmupBaseline(baseline, opts.bl_warmup_epochs, warmup_exp_beta=opts.exp_beta)

    # Load baseline from data, make sure script is called with same type of baseline
    if 'baseline' in load_data:
        baseline.load_state_dict(load_data['baseline'])

    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )

    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)

    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: opts.lr_decay ** epoch)




    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        # Set the random states
        # Dumping of state was done before epoch callback, so do that now (model is loaded)
        baseline.epoch_callback(model, epoch_resume)
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    if opts.eval_only:
        validate(model, val_dataset, opts)
    else:
        print("Start training, lr={} for run {}".format(optimizer.param_groups[0]['lr'], opts.run_name))
        for epoch in range(opts.epoch_start, opts.n_epochs):
            train_epoch(
                model,
                optimizer,
                baseline,
                lr_scheduler,
                epoch,
                val_dataset,
                problem,
                tb_logger,
                opts
            )


if __name__ == "__main__":
    run(get_options())
