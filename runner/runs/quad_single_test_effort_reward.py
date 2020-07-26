from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [1111, 2222, 3333]),
    ('rollout', [128]),
    ('recurrence', [1]),
    ('batch_size', [1024]),
    ('nonlinearity', ['tanh']),
    ('actor_critic_share_weights', ['False']),
    ('policy_initialization', ['xavier_uniform']),
    ('adaptive_stddev', ['False']),
    ('hidden_size', [64]),
    ('with_vtrace', ['False']),
    ('max_policy_lag', [100000000]),
    ('gae_lambda', [1.00]),
    ('max_grad_norm', [0.0]),
    ('entropy_loss_coeff', [0.0]),
    ('quads_effort_reward', [0.0, 0.01, 0.05]),
])

_experiment = Experiment(
    'quads_gridsearch',
    'python -m run_algorithm --env=quadrotor_single --train_for_env_steps=1000000000 --algo=APPO --use_rnn=False --num_workers=36 --num_envs_per_worker=4',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quad_single_test_effort_reward_v102_3seeds', experiments=[_experiment])

# Add the command
# xvfb-run python -m runner.run --run=quad_single_test_effort_reward --runner=processes --max_parallel=9  --pause_between=1 --experiments_per_gpu=3 --num_gpus=4

# Based on launch/train_quadrotor_single_mlp.sh
# python -m algorithms.appo.train_appo --env=quadrotor_single \
#  --train_for_seconds=3600000 \
#  --algo=APPO \
#  --gamma=0.99 \
#  --use_rnn=False \
#  --num_workers=20 \
#  --num_envs_per_worker=8 \
#  --num_policies=1 \
#  --ppo_epochs=1 \
#  --rollout=128 \
#  --recurrence=1 \
#  --batch_size=1024 \
#  --nonlinearity=tanh \
#  --actor_critic_share_weights=False \
#  --policy_initialization=xavier_uniform \
#  --adaptive_stddev=False \
#  --hidden_size=64 \
#  --with_vtrace=False \
#  --max_policy_lag=100000000 \
#  --gae_lambda=1.00 \
#  --max_grad_norm=0.0 \
#  --experiment=quads_single_v102 \
#  --entropy_loss_coeff=0.0