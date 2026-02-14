# HMB Contraception Model

A modeling framework for analyzing the impact of hormonal IUD, oral contraceptive pills, and tranexamic acid on heavy menstrual bleeding (HMB), which in turn affects schooling attendance and anemia outcomes.

## Overview

This package extends [Starsim](https://github.com/starsimhub/starsim) and [FPsim](https://github.com/fpsim/fpsim) to model heavy menstrual bleeding and interventions to reduce it. The model tracks:

- Heavy menstrual bleeding states and transitions
- Contraceptive use (hormonal IUD, oral pills)
- Tranexamic acid treatment
- Educational impacts of HMB
- Population-level outcomes

## Installation

### Quick install

Clone the repository and install in development mode:

```bash
git clone https://github.com/[your-org]/HMB_contraception.git
cd HMB_contraception
pip install -e .
```

### Install with all dependencies

```bash
pip install -e .[dev]
```

## Requirements

- Python 3.9-3.13
- starsim
- fpsim
- numpy
- pandas
- sciris
- matplotlib
- seaborn

## Usage

### Basic example

```python
import hmb_contraception as hc
import starsim as ss

# Create a simulation with HMB module
sim = ss.Sim(
    modules=[
        hc.Menstruation(),
        hc.Education(),
    ]
)
sim.run()
sim.plot()
```

### Running scenarios

The package includes several example scripts:

- `test_run.py` - Basic test of the model
- `run_kenya.py` - Kenya-specific parameterization
- `run_kenya_package_extended.py` - Extended intervention package analysis
- `run_sensitivity_analysis.py` - Parameter sensitivity analysis

## Project structure

- `menstruation.py` - Core HMB state module
- `interventions.py` - Contraceptive and treatment interventions
- `education.py` - Educational impact modeling
- `data/` - Input data files
- `figures/` - Output visualizations
- `results_stochastic_extended/` - Saved simulation results

## Version

Current version: 0.1.0

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

[Add license information]

## Citation

[Add citation information when available]

## Contact

[Add contact information]
