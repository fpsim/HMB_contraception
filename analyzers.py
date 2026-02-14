"""
Analyzers for HMB intervention package
"""

import numpy as np
import starsim as ss
import sciris as sc


class track_care_seeking(ss.Analyzer):
    """
    Analyzer to track care-seeking prevalence stratified by anemia, pain, HMB status, and age.

    This analyzer tracks who is seeking care each month among different population subgroups:
    - Overall care-seeking prevalence
    - By anemia status (anemic vs not anemic)
    - By pain status (pain vs no pain)
    - By HMB status (hmb vs no hmb)
    - By age group (15-24, 25-34, 35-44, 45+)
    - By combinations (e.g., anemic + pain, anemic + hmb, etc.)

    The analyzer requires:
    - sim.people.menstruation with states: anemic, pain, hmb, menstruating
    - sim.interventions.hmb_care_pathway with state: is_seeking_care
    """

    def __init__(self, age_bins=None, **kwargs):
        """
        Initialize the care-seeking analyzer.

        Args:
            age_bins (list): Age bin edges for stratification (default: [15, 25, 35, 45, 100])
        """
        super().__init__(**kwargs)

        # Define age bins
        if age_bins is None:
            age_bins = [15, 25, 35, 45, 100]
        self.age_bins = age_bins
        self.n_age_bins = len(age_bins) - 1

        return

    def init_results(self):
        """Initialize results storage"""
        super().init_results()

        # Build list of results
        results = [
            # Overall care-seeking prevalence
            ss.Result('care_seeking_prev', scale=False, label="Care-seeking prevalence (all menstruating)"),

            # Stratified by single conditions
            ss.Result('care_seeking_anemic', scale=False, label="Care-seeking prevalence (anemic)"),
            ss.Result('care_seeking_not_anemic', scale=False, label="Care-seeking prevalence (not anemic)"),
            ss.Result('care_seeking_pain', scale=False, label="Care-seeking prevalence (pain)"),
            ss.Result('care_seeking_no_pain', scale=False, label="Care-seeking prevalence (no pain)"),
            ss.Result('care_seeking_hmb', scale=False, label="Care-seeking prevalence (HMB)"),
            ss.Result('care_seeking_no_hmb', scale=False, label="Care-seeking prevalence (no HMB)"),

            # Stratified by combinations
            ss.Result('care_seeking_anemic_pain', scale=False, label="Care-seeking prevalence (anemic + pain)"),
            ss.Result('care_seeking_anemic_hmb', scale=False, label="Care-seeking prevalence (anemic + HMB)"),
            ss.Result('care_seeking_pain_hmb', scale=False, label="Care-seeking prevalence (pain + HMB)"),
            ss.Result('care_seeking_all_three', scale=False, label="Care-seeking prevalence (anemic + pain + HMB)"),
        ]

        # Add age-stratified results
        for i in range(self.n_age_bins):
            age_label = f"{int(self.age_bins[i])}-{int(self.age_bins[i+1])-1}"
            if self.age_bins[i+1] >= 100:
                age_label = f"{int(self.age_bins[i])}+"
            results.append(ss.Result(f'care_seeking_age_{i}', scale=False, label=f"Care-seeking prevalence (age {age_label})"))

        # Define all results at once
        self.define_results(*results)

        return

    def step(self):
        """Calculate care-seeking prevalence for each group at this timestep"""
        sim = self.sim
        ti = self.ti

        # Get references to the relevant states
        menstruation = sim.people.menstruation

        # Find the HMB care pathway intervention, or return if not found
        pathway  = None
        for intv in sim.interventions.values():
            if hasattr(intv, 'is_seeking_care'):
                pathway = intv
                break
        if pathway is None:
            return

        # Get the states we need
        menstruating = menstruation.menstruating
        anemic = menstruation.anemic
        pain = menstruation.pain
        hmb = menstruation.hmb
        seeking_care = pathway.is_seeking_care
        age = sim.people.age

        # Helper function to calculate prevalence safely
        def calc_prev(seeking, population):
            """Calculate prevalence, returning 0 if population is empty"""
            n_pop = np.count_nonzero(population)
            if n_pop > 0:
                return np.count_nonzero(seeking & population) / n_pop
            else:
                return 0.0

        # Overall care-seeking among menstruating women
        self.results.care_seeking_prev[ti] = calc_prev(seeking_care, menstruating)

        # By anemia status
        anemic_menstruating = anemic & menstruating
        not_anemic_menstruating = ~anemic & menstruating
        self.results.care_seeking_anemic[ti] = calc_prev(seeking_care, anemic_menstruating)
        self.results.care_seeking_not_anemic[ti] = calc_prev(seeking_care, not_anemic_menstruating)

        # By pain status
        pain_menstruating = pain & menstruating
        no_pain_menstruating = ~pain & menstruating
        self.results.care_seeking_pain[ti] = calc_prev(seeking_care, pain_menstruating)
        self.results.care_seeking_no_pain[ti] = calc_prev(seeking_care, no_pain_menstruating)

        # By HMB status
        hmb_menstruating = hmb & menstruating
        no_hmb_menstruating = ~hmb & menstruating
        self.results.care_seeking_hmb[ti] = calc_prev(seeking_care, hmb_menstruating)
        self.results.care_seeking_no_hmb[ti] = calc_prev(seeking_care, no_hmb_menstruating)

        # By combinations
        anemic_pain = anemic & pain & menstruating
        anemic_hmb = anemic & hmb & menstruating
        pain_hmb = pain & hmb & menstruating
        all_three = anemic & pain & hmb & menstruating

        self.results.care_seeking_anemic_pain[ti] = calc_prev(seeking_care, anemic_pain)
        self.results.care_seeking_anemic_hmb[ti] = calc_prev(seeking_care, anemic_hmb)
        self.results.care_seeking_pain_hmb[ti] = calc_prev(seeking_care, pain_hmb)
        self.results.care_seeking_all_three[ti] = calc_prev(seeking_care, all_three)

        # By age group
        for i in range(self.n_age_bins):
            age_mask = (age >= self.age_bins[i]) & (age < self.age_bins[i+1])
            age_menstruating = age_mask & menstruating
            self.results[f'care_seeking_age_{i}'][ti] = calc_prev(seeking_care, age_menstruating)

        return
