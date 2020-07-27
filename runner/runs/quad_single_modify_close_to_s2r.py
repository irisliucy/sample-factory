from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [1111, 2222, 3333]),
    ('ppo_clip_ratio', [0.05]),
    ('actor_critic_share_weights', ['False']),
    ('nonlinearity', ['tanh']),
    ('policy_initialization', ['xavier_uniform']),
    ('adaptive_stddev', ['False']),
    ('entropy_loss_coeff', [0.0]),
    ('with_vtrace', ['False']),
    ('hidden_size', [64]),
    ('max_policy_lag', [100000000]),
    ('gae_lambda', [1.00]),
    ('max_grad_norm', [0.0]),
    ('recurrence', [1]),

    ('batch_size', [1400]),
    ('rollout', [700]),
    ('num_batches_per_iteration', [1]),
    ('learning_rate', [0.001]),
    ('ppo_epochs', [1, 4, 8, 16]),
])

_experiment = Experiment(
    'quads_gridsearch',
    'python -m run_algorithm --env=quadrotor_single --train_for_env_steps=1000000000 --algo=APPO --use_rnn=False --num_workers=36 --num_envs_per_worker=4',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_single_modify_close_to_s2r_v102_3seeds', experiments=[_experiment])

# Based on https://github.com/amolchanov86/quad_sim2multireal/blob/master/quad_train/config/ppo__crazyflie_baseline.yml
# seed: 1
# variant:
#   env: QuadrotorEnv
#   env_param:
#     dynamics_params: crazyflie
#     dynamics_change:
#       noise:
#         thrust_noise_ratio: 0.05
#       damp:
#         vel: 0.
#         omega_quadratic: 0.
#     init_random_state: True
#     sim_freq: 200 #Hz
#     sim_steps: 2
#     ep_time: 7
#     sense_noise: default
#     rew_coeff:
#       pos: 1.
#       pos_log_weight: 0.
#       pos_linear_weight: 1.
#       effort: 0.05
#       spin: 0.1
#       vel: 0.0
#       crash: 1.
#       orient: 1.
#       yaw: 0.
#   alg_class: PPO
#   alg_param:
#     batch_size: 28000
#     max_path_length: 700
#     n_itr: 3000 #Max num of iterations
#     max_samples: 10000000000 #Max num of samples
#     discount: 0.99
#     step_size: 0.01
#     clip_range: 0.05
#     optimizer_args:
#       batch_size: 128
#       max_epochs: 20
#     plot: False
#     store_paths: False
#     play_every_itr: null
#     record_every_itr: 100
#   baseline_class: GaussianMLPBaseline
#   baseline_param: {}
#   policy_class: GaussianMLPPolicy
#   policy_param:
#     hidden_sizes: [64, 64]
