"""
Education module
"""

import starsim as ss
import numpy as np
import sciris as sc
import pandas as pd


class SocioDemographic(ss.Module):
    def __init__(self, name='sd'): 
        super().__init__(name=name)
        self.define_states(ss.BoolState('urban', default=ss.bernoulli(p=0.5)))
        return

    def init_results(self):
        """ Initialize results """
        super().init_results()
        self.define_results(ss.Result('prop_urban', label='Urban proportion', scale=False))
        return

    def step(self): pass

    def update_results(self):
        """ Update results for socio-demographic module """
        self.results.prop_urban[self.ti] = np.count_nonzero(self.urban) / len(self.urban)
        return


class Education(ss.Module):
    def __init__(self, pars=None, objective_data=None, attainment_data=None, name='edu', **kwargs):
        super().__init__(name=name)

        # Define parameters
        self.define_pars(
            age_start=6,  # Age at which education starts
            age_stop=25,  # Age at which education stops - assumption
            dropout_pars=sc.objdict(  # Parameters for determining dropout probabilities
                intercept=-3,  # Baseline probability of dropping out for girls 10-14
                age_15_19=1.1,  # Adjustment for ages 15-19
                age_20_24=-2,  # Adjustment for ages 20 and older
                parity=1.,  # Adjustment for parity
            ),
            init_dropout=ss.bernoulli(p=0.5),  # Initial dropout probability
            disrupt_pars=sc.objdict(  # Parameters for determining disruption probabilities
                    base=0.01,        # Baseline disruption when no HMB
                    hmb=2.6,          # Strong effect of HMB on disruption
                    hiud=-1.0,        # Effect of hormonal IUD - reduces disruption
                    pill=-1.0,        # Effect of pill - reduces disruption
                    txa=-0.8,         # Effect of TXA - reduces disruption
                    nsaid=-0.5,       # Effect of NSAID - reduces disruption
                    ),
            init_disrupt=ss.bernoulli(p=0.5), # Initial disruption probability
        )
        self.update_pars(pars, **kwargs)

        # Probabilities of dropping out - calculated using data inputs
        self._p_dropout = ss.bernoulli(p=0)

        # Probabilities of disruption of schooling
        self._p_disrupt = ss.bernoulli(p=0)


        # Define states
        self.define_states(
            ss.FloatArr('objective', default=self.get_obj_dist(objective_data)),  # Education objectives
            ss.FloatArr('attainment', default=0),  # Education attainment - initialized as 0, reset if data provided
            ss.BoolState('started', default=False),  # Whether education has been started
            ss.BoolState('in_school'),  # Currently in school
            ss.BoolState('completed'),  # Whether education is completed
            ss.BoolState('dropped'),  # Whether education was dropped
            ss.BoolState('disrupted'), # Whether schooling was disrupted for partial timestep
        )

        # Store things that will be processed after sim initialization
        self.attainment_data = attainment_data

        return

    @property
    def age_15_19(self):
        return (self.sim.people.age >= 15) & (self.sim.people.age < 20)

    @property
    def age_20_24(self):
        return (self.sim.people.age >= 20) & (self.sim.people.age < 25)

    @property
    def parity(self):
        return self.sim.people.parity

    @property
    def hmb(self):
        return self.sim.people.menstruation.hmb

    @staticmethod
    def get_obj_dist(objective_data):
        """
        Return an educational objective distribution based on provided data.
        The data should be provided in the form of a pandas DataFrame with
         "edu" and "percent" as columns.
        Returns:
            An ``ss.Dist`` instance that returns an educational objective for newly created agents
        """
        if objective_data is None:
            dist = ss.uniform(low=0, high=24, name='Educational objective distribution')
        else:
            # Process
            if isinstance(objective_data, pd.DataFrame):
                # Check whether urban is a column
                if 'urban' in objective_data.columns:
                    objective_data = objective_data[objective_data['urban'] == True]  # TODO temp
                bins = objective_data['edu'].values
                props = objective_data['percent'].values

            # Convert to a histogram
            dist = ss.histogram(values=props, bins=bins, name='Educational objective distribution')

        return dist

    def init_attainment(self):
        """
        Initialize educational attainment based on attainment data, if provided.
        """
        if self.attainment_data is None:
            return
        else:
            ppl = self.sim.people
            f_uids = self.sim.people.female.uids
            if isinstance(self.attainment_data, pd.DataFrame):
                edu = self.attainment_data['edu'].values
                f_ages = np.floor(ppl.age[f_uids]).astype(int)
                self.attainment[f_uids] = np.floor(edu[f_ages])
            prev_started = self.attainment > 0
            self.started[prev_started] = True

            # Figure out who's completed their education
            completed_uids = self.attainment >= self.objective
            self.completed[completed_uids] = True

            # Figure out who's dropped out
            past_school_age = ((self.attainment < self.objective)
                                & (ppl.age >= self.pars.age_stop)
                                & ppl.female
                                & self.started)
            self.dropped[past_school_age] = True

            # Figure out who's still in school
            incomplete = ((self.attainment < self.objective)
                        & (ppl.age < self.pars.age_stop)
                        & (ppl.age >= self.pars.age_start)
                        & self.sim.people.female
                        & self.started
                        & ~self.dropped)
            dropped, in_school = self.pars.init_dropout.split(incomplete.uids)
            self.dropped[dropped] = True
            self.in_school[in_school] = True

        return

    def init_results(self):
        """ Initialize results """
        super().init_results()
        results = [
            ss.Result('mean_attainment', label='AGYW: mean education attainment', scale=False),
            ss.Result('mean_objective', label='AGYW: mean education objective', scale=False),
            ss.Result('prop_completed', label='AGYW: proportion completed education', scale=False),
            ss.Result('prop_in_school', label='AGYW: proportion in school', scale=False),
            ss.Result('prop_dropped', label='AGYW: proportion dropped', scale=False),
            ss.Result('prop_disrupted', label='AGYW: proportion disrupted this timestep', scale=False),
            ss.Result('n_disruptions', label='AGYW: cumulative disruptions', scale=False),
        ]
        self.define_results(*results)
        return

    def init_post(self):
        """ Initialize with sim properties """
        super().init_post()
        self.init_attainment()
        return

    def step(self):
        self.start_education()  # Start school
        self.process_disruptions() ## school disruption 
        self.advance_education()  # Advance attainment, determine who reaches their objective, process dropouts
        self.process_dropouts()  # Process dropouts
        self.graduate()  # Check if anyone achieves their education goal
        return

    def start_education(self):
        """
        Begin education. TODO, this assumes everyone starts, but in reality, some may not start school or start later
        """
        new_students = (~self.in_school & ~self.completed & ~self.dropped
                        & (self.sim.people.age >= self.pars["age_start"])
                        & (self.sim.people.age < self.pars["age_stop"])
                        & self.sim.people.female
                        )
        self.in_school[new_students] = True
        self.started[new_students] = True  # Track who started education
        return

    def process_disruptions(self):
        """
        Process schooling disruptions based on HMB status with probability.
        Students with HMB have a probability of experiencing disruption each timestep.
        """
        # Reset disruption status for this timestep
        self.disrupted[:] = False
        
        # Get the uids of individuals currently in school
        uids = self.in_school.uids
        
        if len(uids) == 0:
            return
    
        
        p = self.pars.disrupt_pars
        rhs = np.full_like(uids, fill_value=-np.log(1 / p.base - 1), dtype=float)
    
        # Add covariates 
        for term, val in p.items():
            if term != 'base':
                # For HMB, use local property
                if term == 'hmb':
                    rhs += val * getattr(self, term)[uids]
                # For interventions (hiud, pill, txa, nsaid), get from menstruation module
                else:
                    if hasattr(self.sim.connectors, 'menstruation') and hasattr(self.sim.connectors.menstruation, term):
                        rhs += val * getattr(self.sim.connectors.menstruation, term)[uids]
    
        # Calculate probability - scale by dt for monthly timesteps
        p_val = self.t.dt_year * (1 / (1 + np.exp(-rhs)))
    
        # Apply probability filter
        self._p_disrupt.set(p_val)
        is_disrupted = self._p_disrupt.filter(uids)
    
        # Set disruption status (but keep them in_school)
        self.disrupted[is_disrupted] = True
        
        return
    
    def advance_education(self):
        """
        Increment education attainment
        Students who are disrupted this month have reduced attainment
        """
        students = self.in_school
        
        # Full attainment for non-disrupted students
        not_disrupted = students & ~self.disrupted
        self.attainment[not_disrupted] += self.t.dt_year
        
        # Reduced attainment for disrupted students (e.g., 50% of normal)
        disrupted = students & self.disrupted
        self.attainment[disrupted] += 0.5 * self.t.dt_year

        return

    def process_dropouts(self):
        uids = self.in_school.uids
        p = self.pars.dropout_pars
        rhs = np.full_like(uids, fill_value=p.intercept, dtype=float)

        # Covariates for model prediction
        for term, val in p.items():
            if term != 'intercept':
                rhs += val * getattr(self, term)[uids]

        # Calculate the probability - scale by dt
        p_val = self.t.dt_year * (1 / (1+np.exp(-rhs)))
        self._p_dropout.set(0)
        self._p_dropout.set(p_val)
        drops_out = self._p_dropout.filter(uids)
        self.in_school[drops_out] = False
        self.dropped[drops_out] = True
        return

    def graduate(self):
        completed = self.attainment >= self.objective
        self.in_school[completed.uids] = False
        self.completed[completed.uids] = True
        return

    def update_results(self):
        """ Update results for education module """
        ppl = self.sim.people
        agyw = ppl.female & (ppl.age >= 13) & (ppl.age < 25)  # Adolescent girls & young women
        self.results.mean_attainment[self.ti] = np.mean(self.attainment[agyw])
        self.results.mean_objective[self.ti] = np.mean(self.objective[agyw])
        self.results.prop_completed[self.ti] = np.count_nonzero(self.completed[agyw]) / len(self.completed[agyw])
        self.results.prop_in_school[self.ti] = np.count_nonzero(self.in_school[agyw]) / len(self.in_school[agyw])
        self.results.prop_dropped[self.ti] = np.count_nonzero(self.dropped[agyw]) / len(self.dropped[agyw])
        
        # HMB disruption results
        agyw_in_school = agyw & self.in_school
        if np.any(agyw_in_school):
            self.results.prop_disrupted[self.ti] = np.count_nonzero(self.disrupted[agyw_in_school]) / np.count_nonzero(agyw_in_school)
        else:
            self.results.prop_disrupted[self.ti] = 0

        # Cumulative disruptions (running total of disruption events among AGYW over time)
        # Track across timesteps by accumulating the number disrupted each timestep
        if self.ti == 0:
            # Initialize accumulator on first timestep
            self._cumulative_disruptions = 0
        current_disruptions = np.count_nonzero(self.disrupted[agyw])
        self._cumulative_disruptions += current_disruptions
        self.results.n_disruptions[self.ti] = self._cumulative_disruptions

        return
