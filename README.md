# RelAIBotiX <!-- omit from toc -->
Dynamic Reliability Assessment Framework for AI-Controlled Robotic Systems

## Table of Contents <!-- omit from toc -->
- [Introduction](#introduction)
- [Features](#features)
- [Approach](#approach)
- [Installation](#installation)

## Introduction
AI-controlled robotic systems can introduce significant risks to both humans and the environment. Traditional reliability assessment methods fall short in addressing the complexities of these systems, particularly when dealing with black-box or dynamically changing control policies. These traditional approaches are applied manually and do not consider frequent software updates. 

RelAIBotiX presents a new methodology that enables dynamic and continuous reliability assessment, specifically tailored for robotic systems controlled by AI algorithms.

## Features
- **Time-Series Data Collection**: Supports real and simulated robotic systems
- **Reliability Analysis**: Evaluates control policies based on recorded time-series data
- **Skill Detection & Behavioral Analysis**: Identifies robotic behaviors and their impact on reliability
- **Reliability Model Generation**: Constructs probabilistic models such as fault trees and Markov chains
- **Decision Support**: Provides insights on whether a control policy should be deployed in a real system

## Approach
![RelAIBotiX Framework](figures/relaibotix_new.png)
The application of the RelAIBotiX framework requires three key steps:

### 1. Data Generation
To utilize the framework, time-series data must be collected from either a real robotic system or its corresponding simulation. The recorded data may include:
- Position, velocity, and torque of the gripper
- Position, velocity, and torque of the joints
- Cartesian coordinates of the manipulator and its orientation

### 2. Reliability Assessment
RelAIBotiX performs a comprehensive reliability assessment based on the collected time-series data. This assessment consists of four key methods:
- **Skill Detector**: Identifies robotic skills from time-series data
- **Behavioral Analysis**: Evaluates how system behavior impacts reliability
- **Reliability Model Generator**: Constructs reliability models such as Markov chains and fault trees
- **Reliability Model Solver**: Computes failure probabilities and performs sensitivity analysis

The result is a detailed reliability report, which includes:
- Overall system reliability
- Probability of failure for individual skills
- Sensitivity analysis of critical components

### 3. Decision Making
Using the generated reliability report, an informed decision can be made regarding the deployment of a control policy on the real robotic system. If the reliability assessment indicates potential risks, adjustments can be made before execution to improve the safety and robustness of the policy.

## Installation

```bash
git clone https://github.com/your-username/RelAIBotiX.git
cd RelAIBotiX
pip install -r requirements.txt
```

## Usage
Provide instructions on how to run the framework, including data input format and execution steps.

## License
This project is licensed under the MIT License.

