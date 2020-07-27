from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [1111, 2222, 3333, 4444]),
    ('recurrence', [1]),
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
    ('rollout', [128]),
    ('batch_size', [1024]),
    ('learning_rate', [0.001]),
])

_experiment = Experiment(
    'quads_gridsearch',
    'python -m run_algorithm --env=quadrotor_single --train_for_env_steps=1000000000 --algo=APPO --use_rnn=False --num_workers=72 --num_envs_per_worker=6',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_single_based_on_lunch_mlp_lr_0.001_v102_4seeds', experiments=[_experiment])

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
