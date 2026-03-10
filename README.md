# Dynasto

This repository contains the implementation of the Dynasto approach, described in the paper "Dynasto: Validity-Aware Dynamic–Static Parameter Optimization
  for Autonomous Driving Testing". This project focuses on training adversarial agents to test autonomous driving policies in a custom multi-agent environment based on [Highway-Env](https://highway-env.farama.org/environments/highway/). The adversaries learn to generate challenging scenarios for various ego agent configurations, with the goal of improving the robustness of autonomous driving systems.


# Purpose
This artifact provides the implementation of the Dynasto approach, including adversarial RL-based agent training with further improvement of test sceanrios with a genetic algorithm. This artifact also includes the supporting scripts, such as ego agent training, execution trace analysis, results post processing, and visualization.
We apply for all the three badges: available, reusable and functional.

**Available.**
The complete artifact is publicly available and includes all the source code, scripts, configuration files, and instructions required to run the experiments presented in the paper. The repository provides the implementation of the Dynasto approach, including adversarial RL-based agent training, test scenario improvement using a genetic algorithm, and supporting utilities such as ego agent training, execution trace analysis, results post-processing, and visualization.

**Functional.**
The artifact has been tested to ensure that the provided scripts execute correctly and reproduce the main steps of the experimental pipeline described in the paper. Detailed setup instructions and execution commands are provided to allow evaluators to run the adversarial training process, perform scenario optimization, and generate the corresponding analysis and results.

**Reusable.**
The artifact is designed to facilitate reuse and extension by other researchers. The codebase is modular, with clearly separated components for reinforcement learning, scenario optimization, and analysis.
# Setup 

## Installation
It is recommended to use a virtual environment. We recommend using conda for managing the environment, but you can also use venv or any other virtual environment tool. Below are the steps to set up the environment and install the necessary dependencies.

The software has been tested on Python 3.10, and we recommend using this version for compatibility. The software was tested with Windows oprating system, but it should also work on Linux and macOS.

1) Create and activate a venv (example):
```python
conda create -n dynasto python=3.10
conda activate dynasto
```

2) Install core dependencies:
```python
pip install .
```
if you plan editing the codebase, you can install the package in editable mode:
```python
pip install -e .
```

3) During the installation Pytorch will be installed. If you want to use Pytorch with GPU, make sure to install the correct version of Pytorch for your system and CUDA version. You can find the appropriate command for your setup on the [PyTorch website](https://pytorch.org/get-started/locally/). No extra installation is needed if you are using CPU.


## Usage

In this section, we provide instructions on how to use the codebase to train ego agents, adversarial agents, and analyze results. The code is organized into several scripts, each responsible for a specific part of the process.

### Training ego-agents

To train an ego agent (the system under test), execute the following command:

```python
python ego_agent_training\\train_ego_agent.py
```
You can specify different configurations for the ego agent by modifying the `ego_config` dictionary in the `src/dynasto/configs/agent_configs.py` file. The training process will save the trained model and training logs for later evaluation.

To set the adversarial vehicles that wil be used during the ego agent training, you can modify the `other_vehicles` parameter in the `ego_config` dictionary. Available options include: `highway_env.vehicle.behavior.DefensiveVehicle`, `highway_env.vehicle.behavior.AggressiveVehicle`, and `highway_env.vehicle.behavior.IDMVehicle`. Defensive is specified by default.

You can specify the number of training episodes and the duration of each episode by modifying the `EPISODES` variable. In our experiments, we set it to 4000, but for demo purposes, you can set it to a smaller number (e.g., 40) to see the training process in action.

To enable or disable rendering as well as intermediary checkpoint saving, you can modify the `RenderCallback` and `CheckpointCallback` parameters in the `train_ego.py` script.

By default, rendering is enabled and checkpoints are saved every 500 steps. 

To evaluate the trained agent and visualize the execution traces, execute the following command:

```python
ego_agent_training\\test_ego.py
``` 

### Randomized and genetic algorithm-based adversarial agent 
To train a randomized adversarial agent, change te `algo` variable in the `adversarial_test_generation\\train_adversary.py` script to `random` and execute the following command:

```python
python adversarial_test_generation\\train_adversary.py 
```

To train a genetic algorithm-based adversarial agent, change the `algo` variable to `ga` and execute the same command.
Genetic algorithm parameters can be modified in the `adversarial_test_generation\\tester_config.yaml` file. The parameters include population size, number of generations, mutation probability, and others.

The exection results will be saved in the `stats` folder, and the trained models will be saved in the `weights` folder. 
We provide the pre-trained models for the ego vehicles, which can be selected by modifying the `ego_type` variable in the `train_adversary.py` script. Available options include `use_case_1` and `use_case_2`.

### Training RL-based adversarial agent 
To train a reinforcement learning-based adversarial agent, change the `algo` variable in the `adversarial_test_generation\\train_adversary.py` script to `dqn` and execute the following command:

```python
python adversarial_test_generation\\train_adversary.py 
```

### Training co-evoluanary adversarial agent
To run the co-evolutionary setup, where the initialization of the test scenarios is optimized with a genetic algorithm while the adversarial agent is trained with reinforcement learning, execute the following command:

```python
python adversarial_test_generation\\coevolutionary_setup.py 
```

### Training Dynasto agent

To train the Dynasto agent, which combines reinforcement learning and genetic algorithm for test scenario generation, execute the following command:

```python
python adversarial_test_generation\\dynasto_setup.py 
```

### Results analysis and visualization

#### Plotting rewards

For the demo purposes, execute the following command to obtain the execution traces (number of episodes is set to 100 for a quick demo and the number of runs is set to 3):
```python
python adversarial_test_generation\\train_adversary.py 
```

You will see the traces saved in the `stats` folder. Each run contains the general information about the execution, such as average reward and crash rate in `adv_vehicle_stats` files (csv and json files have the same content). Each failure is saved in a separate folder with a name corresponding to the id of the failure. Similar i.e. almost identical failures will be grouped together in the same folder. 

As the first step of analysis plot the rewards and crash rates for the trained adversarial agents by executing the following command:

```python
python .\failure_analysis\plot_rewards.py --results-folder <path_to_results_folder>   
```
Resaults folder contains the folders for each run and can be named, for instantce `stats\rl_<date>-dqn_use_case_1`.
It will plot the average reward, crash rate, true crash rate and number of failures for each run.


#### Comparing algorithms

Once sufficient number of runs is obtained, you can compare the performance of different algorithms by executing the following command:

```python
python failure_analysis\\compare.py --stats-path <path_to_stats_folder> --stats-names <names_for_each_algorithm> --plot-name <name_for_plot>
```
For the demo purposes, execute the following command:

```python
python failure_analysis\\compare.py  --stats-path "extracted_tests\demo\random" "extracted_tests\demo\dqn" --stats-names "random" "dqn"  --plot-name "demo"
```

You will see the plots appear in `stats\\plots` folder. These plots allows to compare the performance of different algorithms in number of failures, number of revealed clusters as well as the diversity of the failures.

For running the the `compare.py` the following files are needed to be present in the folder with execution results:
`convergence_data.json` - contains the convergence data for each run, such as the number of failures at each episode.
`fail_data.json` - total number of failures revealed in each run.
`semantic_failures.json` semantic encoding of each failure revealed in each run. This file is used to calculate the number of revealed clusters and the diversity of the failures.

To obtain these files, run the following commands:

```python
python .\failure_analysis\get_semantic_failures.py --results-folder stats
```

```python
python .\failure_analysis\plot_convergence.py --results-folder stats   
```

After this you will have the necessary files to run the `compare.py` script and compare the performance of different algorithms, whose results are stored in the `stats` folder.

Note, that the algorithms should reveal a sufficient number of failures (at least 10) to be able to compare them. We thus provided pre-recorded results for the demo purposes in the `extracted_tests\demo` folder, which contains the results for the random and DQN-based adversarial agents.

## Questions and support
If you have any questions or need support regarding the setup, usage, or any aspect of the codebase, please feel free to reach out:  
Dmytro Humeniuk gumenyuk98@gmail.com