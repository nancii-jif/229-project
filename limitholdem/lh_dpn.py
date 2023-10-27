'''copy pasting from run_rl.py
'''
import os 
import argparse

import torch

import rlcard 
from rlcard.agents import RandomAgent
from rlard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve
)

def train(args):

    device = get_device()
    set_seed(args.seed)

    env = rlcard.make(
        args.env,
        config={
            'seed': args.seed,
        }
    )

    if args.algorithm == 'dqn':
        from rlcard.agents import DQNAgent
        if args.load_checkpoint_path != "":
            agenet = DQNAgent.fromt_checkpoint(checkpoint=torch.load(args.load_checkpoint_path))
        else:
            agent = DQNAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=[64, 64],
                device=device,
                save_path=args.log_dir,
                save_every=args.save_every
            )

    #else if 'nfsp'
    elif args.algorithm == 'nfsp':
         from rlcard.agents import NSFPAgent
         if args.load_checkpoint_path != "":
             agent = NFSPAgent.from_checkpoint(checkpoint=torch.load(args.load_checkpoint_path))
         else:
             agent = NFSPAgent(
                 num_actions = env.num_actions,
                 state_shape=env.state_shape[0],
                 hidden_layers_sizes=[64, 64],
                 q_mlp_layers=[64, 64],
                 device=device,
                 save_path=args.log_dir,
                 save_every=args.save_every
             )
    agent = [agent]
    for _ in range(1, env.num_players):
        agent.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    #training
    with Logger(arg.log_dir) as logger:
        for episode in range(args.num_episodes):
            if args.algorithm == 'nfsp':
                agents[0].sample_episode_policy()

            trajectories, payoffs = env.run(is_training=True)

            trajectories = reorganize(trajectories, payoffs)

            for ts in trajectories[0]:
                agent.feed(ts)

                if episode % args.evaluate_every == 0:
                    logger.log_performance(
                        episode,
                        tournament(
                            env,
                            args.num_eval_games,
                        )[0]
                    )

            csv_path, fig_path = logger.csv_path, logger.fig_path

        plot_curve(csv_path, fig_path, args.algorithm)

        save_path = os.path.join(args.log_dir, 'model.pth')
        torch.save(agent, save_path)
        print('model saved in', save_path)
