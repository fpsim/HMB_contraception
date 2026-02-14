"""
Analyzers for HMB intervention package
"""

import numpy as np
import starsim as ss
import sciris as sc


class track_tx_eff(ss.Analyzer):
    """
    Track treatment effectiveness rates for each treatment type.

    Captures data at the point of assessment (after time_to_assess period)
    to calculate actual efficacy rates before treatments are stopped.

    For each treatment (NSAID, TXA, Pill, hIUD), tracks:
    - Total number of people assessed
    - Number who had effective treatment (HMB resolved)
    - Actual efficacy rate (effective / assessed)

    This analyzer requires sim.interventions.hmb_care_pathway.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Track which UIDs we've already counted to avoid double-counting
        self.counted_uids = {'nsaid': set(), 'txa': set(), 'pill': set(), 'hiud': set()}

        # Cumulative counts
        self.n_assessed = {'nsaid': 0, 'txa': 0, 'pill': 0, 'hiud': 0}
        self.n_effective = {'nsaid': 0, 'txa': 0, 'pill': 0, 'hiud': 0}

        return

    def step(self):
        """Track treatment effectiveness at each timestep"""

        # Find the HMB care pathway intervention
        pathway = None
        for intv in self.sim.interventions.values():
            if hasattr(intv, 'treatment_map'):
                pathway = intv
                break

        if pathway is None:
            return

        # Get UIDs of people who have been assessed
        assessed_uids = pathway.treatment_assessed.uids

        # For each treatment type, count new assessments
        for tx_name, tx_idx in pathway.treatment_map.items():
            if tx_name == 'none':
                continue

            # Filter for people on this specific treatment
            on_this_tx = pathway.current_treatment[assessed_uids] == tx_idx
            assessed_this_tx_uids = assessed_uids[on_this_tx]

            # Only count UIDs we haven't counted yet
            new_uids = [uid for uid in assessed_this_tx_uids if uid not in self.counted_uids[tx_name]]

            if len(new_uids) > 0:
                # Convert to numpy array for indexing
                new_uids = np.array(new_uids)

                # Add to counted set
                self.counted_uids[tx_name].update(new_uids)

                # Count assessments
                self.n_assessed[tx_name] += len(new_uids)

                # Count how many were effective
                effective = pathway.treatment_effective[new_uids]
                self.n_effective[tx_name] += np.count_nonzero(effective)

        return

    def get_efficacy(self, treatment):
        """
        Calculate actual efficacy rate for a treatment.

        Args:
            treatment: Treatment name ('nsaid', 'txa', 'pill', 'hiud')

        Returns:
            Efficacy rate (n_effective / n_assessed), or None if no assessments
        """
        if self.n_assessed[treatment] > 0:
            return self.n_effective[treatment] / self.n_assessed[treatment]
        else:
            return None


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
