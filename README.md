# RelAIBotiX <!-- omit from toc -->

Dynamic Reliability Assessment Framework for AI-Controlled Robotic Systems

## Table of Contents <!-- omit from toc -->
- [Introduction](#introduction)
- [Approach](#approach)
- [Installation](#installation)
- [Usage](#usage)
  - [Minimal Example](#minimal-example)
  - [Available Commands](#available-commands)
  - [Arguments](#arguments)
- [License](#license)

---

## Introduction
AI-controlled robotic systems can introduce significant risks to both humans and the environment.  
Traditional reliability assessment methods fall short in addressing the complexities of these systems, particularly when dealing with black-box or dynamically changing control policies.  
These traditional approaches are applied manually and do not consider frequent software updates. 

**RelAIBotiX** presents a methodology that enables **dynamic and continuous reliability assessment**, specifically tailored for robotic systems controlled by AI algorithms.

---

## Approach
![RelAIBotiX Framework](artifacts/figures/RelAIBotiX.png)

The framework provides:
- Fault & uncertainty injection in simulation
- Skill detection 
- Behavioral analysis 
- Automatic generation of hybrid reliability models (DTMC, Fault Trees, PRISM)
- Structured reliability reports (CSV, PDF)

---

## Installation

We recommend using a **conda environment** to manage dependencies.

```bash
# Create and activate environment
conda create -n relaibotix python=3.10 -y
conda activate relaibotix

# Clone the repository
git clone https://github.com/your-username/RelAIBotiX.git
cd RelAIBotiX

# Install requirements
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

---

## Usage

After installation, RelAIBotiX is available as a CLI tool.

### Minimal Example

Run the full pipeline on a small dataset:

```bash
relaibotix --h5 datasets/IL/act/act_20.h5
```

This will produce:
- A reliability report (PDF + JSON) in `results/reports/`
- A PRISM model in `results/prism/`

**Note:**  
- Training data demonstrations for imitation learning are provided in  
  `datasets/LeRobot_Demonstrations/`. This folder contains **100 demonstrations** collected for policy training.  
- Additional inference and inference-with-fault scripts are available in  
  `lerobot/scripts/`.

---

### Available Commands

The CLI supports different modes:

- **Full pipeline**

```bash
relaibotix --h5 <dataset.h5> 
```

- **Skill detection only**

```bash
relaibotix_skill_detection --h5 <dataset.h5> --ckpt artifacts/checkpoints/skill_detector.ckpt
```

- **Faulty evaluation only**

```bash
relaibotix_faulty --h5 <dataset_with_faults.h5>
```

---

### Arguments

| Argument       | Description                                     | Default                                      |
|----------------|-------------------------------------------------|----------------------------------------------|
| `--h5`         | Path to the dataset in HDF5 format              | *(required)*                                 |
| `--ckpt`       | Path to skill detector checkpoint               | `artifacts/checkpoints/skill_detector.ckpt`  |
| `--config`     | Path to robot configuration JSON                | `config_files/robots/so_arm_config.json`     |
| `--output`     | Output directory for reports                    | `results/reports`                            |
| `--prism`      | Output directory for PRISM models               | `results/prism`                              |
| `--tol`        | XY tolerance for faulty evaluation [m]          | `0.02`                                       |
| `--target-x`   | Target X position [m]                           | `0.20`                                       |
| `--target-y`   | Target Y position [m]                           | `-0.15`                                      |

---

## License
This project is licensed under the MIT License.