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

            # Filter for people assessed on this specific treatment
            # Use assessed_treatment (which is saved at assessment time) instead of current_treatment
            # (which may be reset to NaN if treatment has stopped)
            on_this_tx = pathway.assessed_treatment[assessed_uids] == tx_idx
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


class track_hmb_anemia(ss.Analyzer):
    """
    Analyzer to track the relationship between HMB and anemia among menstruating non-pregnant women.

    Tracks:
    - Anemia cases among women with/without HMB
    - HMB cases among women with/without anemia
    - Prevalence rates for all combinations

    This analyzer requires:
    - sim.people.menstruation with states: anemic, hmb, menstruating
    - sim.people.fp with state: pregnant (from FPsim)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def init_results(self):
        """Initialize results storage"""
        super().init_results()

        # Define all results
        results = [
            # Counts
            ss.Result('n_anemia_with_hmb', scale=True, label="Anemia cases among women with HMB"),
            ss.Result('n_anemia_without_hmb', scale=True, label="Anemia cases among women without HMB"),
            ss.Result('n_hmb_with_anemia', scale=True, label="HMB cases with anemia"),
            ss.Result('n_hmb_without_anemia', scale=True, label="HMB cases without anemia"),

            # Prevalence rates
            ss.Result('prev_anemia_in_hmb', scale=False, label="Prevalence of anemia among women with HMB"),
            ss.Result('prev_anemia_in_no_hmb', scale=False, label="Prevalence of anemia among women without HMB"),
            ss.Result('prev_hmb_in_anemia', scale=False, label="Prevalence of HMB among women with anemia"),
            ss.Result('prev_hmb_in_no_anemia', scale=False, label="Prevalence of HMB among women without anemia"),

            # Overall population counts
            ss.Result('n_menstruating_nonpreg', scale=True, label="Menstruating non-pregnant women"),
            ss.Result('n_hmb_total', scale=True, label="Total HMB cases"),
            ss.Result('n_anemia_total', scale=True, label="Total anemia cases"),
        ]

        self.define_results(*results)
        return

    def step(self):
        """Calculate HMB-anemia relationships at this timestep"""
        sim = self.sim
        ti = self.ti

        # Get menstruation module
        menstruation = sim.people.menstruation

        # Get states
        menstruating = menstruation.menstruating
        hmb = menstruation.hmb
        anemic = menstruation.anemic

        # Get pregnancy status from FPsim
        pregnant = sim.people.fp.pregnant if hasattr(sim.people, 'fp') else np.zeros(len(sim.people), dtype=bool)

        # Define eligible population: menstruating non-pregnant women
        eligible = menstruating & ~pregnant

        # Helper function to calculate prevalence safely
        def calc_prev(condition, population):
            """Calculate prevalence, returning 0 if population is empty"""
            n_pop = np.count_nonzero(population)
            if n_pop > 0:
                return np.count_nonzero(condition & population) / n_pop
            else:
                return 0.0

        # Overall population counts
        self.results.n_menstruating_nonpreg[ti] = np.count_nonzero(eligible)
        self.results.n_hmb_total[ti] = np.count_nonzero(hmb & eligible)
        self.results.n_anemia_total[ti] = np.count_nonzero(anemic & eligible)

        # Anemia stratified by HMB status
        hmb_eligible = hmb & eligible
        no_hmb_eligible = ~hmb & eligible

        self.results.n_anemia_with_hmb[ti] = np.count_nonzero(anemic & hmb_eligible)
        self.results.n_anemia_without_hmb[ti] = np.count_nonzero(anemic & no_hmb_eligible)

        self.results.prev_anemia_in_hmb[ti] = calc_prev(anemic, hmb_eligible)
        self.results.prev_anemia_in_no_hmb[ti] = calc_prev(anemic, no_hmb_eligible)

        # HMB stratified by anemia status
        anemic_eligible = anemic & eligible
        no_anemic_eligible = ~anemic & eligible

        self.results.n_hmb_with_anemia[ti] = np.count_nonzero(hmb & anemic_eligible)
        self.results.n_hmb_without_anemia[ti] = np.count_nonzero(hmb & no_anemic_eligible)

        self.results.prev_hmb_in_anemia[ti] = calc_prev(hmb, anemic_eligible)
        self.results.prev_hmb_in_no_anemia[ti] = calc_prev(hmb, no_anemic_eligible)

        return


class track_tx_dur(ss.Analyzer):
    """
    Track treatment durations for each treatment type.

    Captures duration at the moment treatment is started to avoid issues
    with durations being overwritten when people switch treatments.

    For each treatment (NSAID, TXA, Pill, hIUD), tracks:
    - List of durations assigned at treatment start

    This analyzer requires sim.interventions.hmb_care_pathway.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Track durations for each treatment
        self.durations = {
            'nsaid': [],
            'txa': [],
            'pill': [],
            'hiud': []
        }

        # Track which UIDs we've already recorded to avoid double-counting
        self.counted_uids = {'nsaid': set(), 'txa': set(), 'pill': set(), 'hiud': set()}

        return

    def step(self):
        """Track treatment durations at each timestep"""

        # Find the HMB care pathway intervention
        pathway = None
        for intv in self.sim.interventions.values():
            if hasattr(intv, 'treatment_map'):
                pathway = intv
                break

        if pathway is None:
            return

        # For each treatment type, check for newly started treatments
        for tx_name, tx_idx in pathway.treatment_map.items():
            if tx_name == 'none':
                continue

            # Get people who accepted this treatment
            accepted = getattr(pathway, f'{tx_name}_accepted')
            accepted_uids = accepted.uids

            # Find new acceptors we haven't counted yet
            new_uids = [uid for uid in accepted_uids if uid not in self.counted_uids[tx_name]]

            if len(new_uids) > 0:
                # Add to counted set
                self.counted_uids[tx_name].update(new_uids)

                # Get their durations (only for NSAID/TXA, as pill/hIUD use FPsim durations)
                if tx_name in ['nsaid', 'txa']:
                    new_uids_arr = np.array(new_uids)
                    durs = pathway.dur_treatment[new_uids_arr]
                    # Only save non-NaN durations
                    valid_durs = durs[~np.isnan(durs)]
                    self.durations[tx_name].extend(valid_durs.tolist())

        return


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
