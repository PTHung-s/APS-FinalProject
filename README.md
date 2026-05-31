# APS Final Project - Advanced Probability & Statistics

This repository contains the solutions for the final project of the **Advanced Probability and Statistics** course.

## Project Description

The project is divided into three main questions:

- **Question 1**: Volume estimation using Monte Carlo simulation and analysis of hit-rate, sample size, and high-dimensional integration.
- **Question 2**: Implementation of the **Metropolis-Hastings** MCMC algorithm.
- **Question 3**: Audio signal entropy analysis to differentiate between **Music** and **Noise**.

## Repository Structure

```bash
APS-FinalProject/
├── Question1/
│   └── Question_1.ipynb
├── Question2/
│   └── Question2.ipynb
├── Question3/
│   ├── 00_convert_mkv_to_wav.py
│   ├── 00_trim_to_9s.py
│   ├── 02_entropy_delta_01.py
│   ├── 03_sensitivity_analysis.py
│   ├── 04_generate_spectrograms.py     # Optional (may cause MemoryError)
│   ├── 05_mutual_information_bonus.py
│   ├── 06_run_question3.py             # Run all Q3 scripts with one command
│   ├── Music/
│   ├── noise/
│   └── results/
├── project_outputs/
├── requirements.txt
└── README.md
```

## Requirements

### System Requirements

* Python 3.9 or higher
* Jupyter Notebook
* FFmpeg for audio conversion in Question 3

### Python Packages

Install all dependencies using:

```bash
pip install -r requirements.txt
```

## How to Run the Code

### 1. Clone the Repository

```bash
git clone https://github.com/PTHung-s/APS-FinalProject.git
cd APS-FinalProject
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Each Question

#### Question 1

```bash
jupyter notebook Question1/Question_1.ipynb
```

#### Question 2

```bash
jupyter notebook Question2/Question2.ipynb
```

#### Question 3: Audio Entropy Analysis

Recommended command:

```bash
cd Question3
python 06_run_question3.py
```

Note for Question 3:

`04_generate_spectrograms.py` is optional. It may cause a `MemoryError` on some machines due to high DPI settings. You can skip this script if it fails because it only generates spectrogram images for visualization.

Manual run:

```bash
python 00_convert_mkv_to_wav.py
python 00_trim_to_9s.py
python 02_entropy_delta_01.py
python 03_sensitivity_analysis.py
python 05_mutual_information_bonus.py
```
