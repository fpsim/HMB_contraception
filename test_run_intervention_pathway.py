import numpy as np
import pandas as pd
import starsim as ss
import fpsim as fp
import sciris as sc

from menstruation import Menstruation
from education import Education
from interventions import HMBCarePathway


# def make_pars():
#     pars = {}
#     pars['exposure_factor'] = 2
#     pars['prob_use_intercept'] = -1
#     return pars

# Create modules
mens = Menstruation()
edu = Education()  # No data files needed for testing

# Create pathway intervention
pathway = HMBCarePathway(
    year=2020,  # Start immediately for testing
    time_to_assess=2,  # Assess after 2 months 
)

# Create simulation
sim = fp.Sim(
    start=2020,
    stop=2025,
    n_agents=1000,
    location='kenya',
    # pars=make_pars(),
    education_module=edu,
    connectors=[mens],
    interventions=[pathway],
    verbose=0.1,
)

sim.run()