from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333, 4444, 5555, 6666, 7777]),
    ('ppo_epochs', [1]),
    ('learning_rate', [0.0001]),
    ('entropy_loss_coeff', [0.0]),
])

_experiment = Experiment(
    'quads_gridsearch',
    'python -m algorithms.appo.train_appo --env=quadrotor_multi --train_for_env_steps=1000000000 --algo=APPO '
    '--gamma=0.99 --use_rnn=False --num_workers=72 --num_envs_per_worker=2 --num_policies=1 --rollout=128 '
    '--recurrence=1 --batch_size=512 --benchmark=False  --with_pbt=False --adam_eps=1e-8 --nonlinearity=tanh '
    '--actor_critic_share_weights=False --policy_initialization=xavier_uniform --adaptive_stddev=False '
    '--hidden_size=64 --with_vtrace=False --max_policy_lag=100000000 --gae_lambda=1.00 --max_grad_norm=0.0 '
    '--ppo_clip_value=5.0',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('quads_multi_v104_8seeds', experiments=[_experiment])
