"""
Run and plot a simulation for Kenya
"""

import pylab as pl
import starsim as ss
import fpsim as fp
import sciris as sc
import pandas as pd
from menstruation import Menstruation
from education import Education


if __name__ == '__main__':

    # Create modules
    mens = Menstruation()
    objective_data = pd.read_csv(f"data/kenya_objective.csv")
    attainment_data = pd.read_csv(f"data/kenya_initialization.csv")
    edu = Education(objective_data=objective_data, attainment_data=attainment_data) 

    # Create sim
    pars = dict(
        n_agents=1000,
        start=2000,
        stop=2032,
        location='kenya',
    )

    sim = fp.Sim(education_module=edu, connectors=[mens], **pars)
    sim.run(verbose=1/12)

    # Plot method mix by age

