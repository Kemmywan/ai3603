Goal:

Episodic Return: 180-250

Episodic Length: 150-250

TD-loss: 0.5-2

Q-Values: 130-150

# 11.7

## E1

python dqn.py --total-timesteps=200000 --learning-rate=0.0001 --buffer-size=50000 --gamma=0.99 --target-network-frequency=500 --start-e=1.0 --end-e=0.01 --exploration-fraction=0.5 --train-frequency=4

LunarLander-v2__dqn__42__1762492639

## E2

python dqn.py --total-timesteps=200000 --learning-rate=0.0001 --buffer-size=50000 --gamma=0.8 --target-network-frequency=500 --start-e=1.0 --end-e=0.01 --exploration-fraction=0.5 --train-frequency=4

LunarLander-v2__dqn__42__1762493202

## E3

python dqn.py --total-timesteps=500000 --learning-rate=0.00005 --buffer-size=100000 --gamma=0.99 --target-network-frequency=500 --start-e=1.0 --end-e=0.01 --exploration-fraction=0.5 --train-frequency=10

LunarLander-v2__dqn__42__1762493900

## E4

Add bias=50 in QNetwork

To avoid negative cycle

python dqn.py --total-timesteps=500000 --learning-rate=0.0001 --buffer-size=50000 --gamma=0.99 --target-network-frequency=500 --start-e=1.0 --end-e=0.01 --exploration-fraction=0.5 --train-frequency=5

LunarLander-v2__dqn__42__1762494391

## E5

python dqn.py --total-timesteps=500000 --learning-rate=0.0001 --buffer-size=50000 --gamma=0.99 --target-network-frequency=500 --start-e=1.0 --end-e=0.01 --exploration-fraction=0.5 --train-frequency=5

LunarLander-v2__dqn__42__1762494893

## E6

python dqn.py --total-timesteps=300000 --learning-rate=0.0001 --buffer-size=50000 --gamma=0.99 --target-network-frequency=500 --start-e=1.0 --end-e=0.05 --exploration-fraction=0.6 --train-frequency=5

LunarLander-v2__dqn__42__1762495399

## E7

python dqn.py --total-timesteps=300000 --learning-rate=0.0001 --buffer-size=50000 --gamma=0.99 --target-network-frequency=1000 --start-e=1.0 --end-e=0.1 --exploration-fraction=0.8 --train-frequency=4

LunarLander-v2__dqn__42__1762495952

# E8

python dqn.py --total-timesteps=300000 --learning-rate=0.0001 --buffer-size=50000 --gamma=0.99 --target-network-frequency=1000 --start-e=1.0 --end-e=0.1 --exploration-fraction=0.8 --train-frequency=4

Add reward-training

LunarLander-v2__dqn__42__1762496685

# E9

Add tau for soft-update

python dqn.py --total-timesteps=300000 --learning-rate=0.0001 --buffer-size=50000 --gamma=0.99 --target-network-frequency=1000 --start-e=1.0 --end-e=0.1 --exploration-fraction=0.8 --train-frequency=4 

LunarLander-v2__dqn__42__1762497321

# E10

python dqn.py --total-timesteps=300000 --learning-rate=0.0001 --buffer-size=10000 --gamma=0.99 --target-network-frequency=1000 --start-e=1.0 --end-e=0.1 --exploration-fraction=0.8 --train-frequency=5 --tau=0.001 

Use exp-Decay

LunarLander-v2__dqn__42__1762497961

# E11

python dqn.py --total-timesteps=300000 --learning-rate=0.0001 --buffer-size=10000 --gamma=0.99 --target-network-frequency=1000 --start-e=1.0 --end-e=0.1 --exploration-fraction=0.8 --train-frequency=5 --tau=0.001 

LunarLander-v2__dqn__42__1762497961

# E12

python dqn.py \
  --total-timesteps=500000 \
  --learning-rate=0.00025 \
  --buffer-size=100000 \
  --gamma=0.99 \
  --tau=0.005 \
  --batch-size=128 \
  --start-e=1.0 \
  --end-e=0.01 \
  --exploration-fraction=0.5 \
  --learning-starts=10000 \
  --train-frequency=4

120-84 -> 256-256

LunarLander-v2__dqn__42__1762498610

# E13

python dqn.py \
  --total-timesteps=10000 \
  --learning-rate=0.00025 \
  --buffer-size=500 \
  --gamma=0.99 \
  --tau=0.005 \
  --batch-size=128 \
  --start-e=1.0 \
  --end-e=0.01 \
  --exploration-fraction=0.5 \
  --learning-starts=100 \
  --train-frequency=4 \
  --target-network-frequency=5

LunarLander-v2__dqn__42__1762499370

# E14

python dqn.py \
  --total-timesteps=100000 \
  --learning-rate=0.00025 \
  --buffer-size=500 \
  --gamma=0.99 \
  --tau=0.005 \
  --batch-size=128 \
  --start-e=1.0 \
  --end-e=0.01 \
  --exploration-fraction=0.5 \
  --learning-starts=100 \
  --train-frequency=4 \
  --target-network-frequency=5

LunarLander-v2__dqn__42__1762499564

# E15

python dqn.py \
  --total-timesteps=100000 \
  --learning-rate=0.00025 \
  --buffer-size=5000 \
  --gamma=0.99 \
  --tau=0.005 \
  --batch-size=128 \
  --start-e=1.0 \
  --end-e=0.01 \
  --exploration-fraction=0.5 \
  --learning-starts=100 \
  --train-frequency=4 \
  --target-network-frequency=5

LunarLander-v2__dqn__42__1762499783

# E16

python dqn.py \
    --total-timesteps=100000 \
    --learning-rate=0.00025 \
    --buffer-size=5000 \
    --gamma=0.99 \
    --tau=0.005 \
    --start-e=1.0 \
    --end-e=0.01 \
    --exploration-fraction=0.8 \
    --learning-starts=5000 \
    --train-frequency=4 \
    --target-network-frequency=50

LunarLander-v2__dqn__42__1762500424

# E17

python dqn.py \
    --total-timesteps=300000 \
    --learning-rate=0.00025 \
    --buffer-size=80000 \
    --gamma=0.99 \
    --tau=0.005 \
    --start-e=1.0 \
    --end-e=0.01 \
    --exploration-fraction=0.8 \
    --learning-starts=10000 \
    --train-frequency=4 \
    --target-network-frequency=500

Finally!!!!A good effect

LunarLander-v2__dqn__42__1762500756

# E18

python dqn.py \
    --total-timesteps=500000 \
    --learning-rate=0.00025 \
    --buffer-size=80000 \
    --gamma=0.99 \
    --tau=0.005 \
    --start-e=1.0 \
    --end-e=0.01 \
    --exploration-fraction=0.7 \
    --learning-starts=10000 \
    --train-frequency=4 \
    --target-network-frequency=1000

LunarLander-v2__dqn__42__1762501359

# E19

python dqn.py \
    --total-timesteps=500000 \
    --learning-rate=0.00025 \
    --buffer-size=80000 \
    --gamma=0.99 \
    --tau=0.005 \
    --start-e=1.0 \
    --end-e=0.01 \
    --exploration-fraction=0.7 \
    --learning-starts=10000 \
    --train-frequency=4 \
    --target-network-frequency=1000

LunarLander-v2__dqn__42__1762507424