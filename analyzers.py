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

        # Find the HMB cascade intervention
        cascade = None
        for intv in self.sim.interventions.values():
            if hasattr(intv, 'treatments'):  # HMBCascade has 'treatments' dict
                cascade = intv
                break

        if cascade is None:
            return

        # For each treatment type, check for new assessments
        for tx_name in ['nsaid', 'txa', 'pill', 'hiud']:
            treatment = cascade.treatments[tx_name]

            # Get UIDs of people who have been assessed for this treatment
            assessed_this_tx_uids = treatment.treatment_assessed.uids

            # Only count UIDs we haven't counted yet
            new_uids = [uid for uid in assessed_this_tx_uids if uid not in self.counted_uids[tx_name]]

            if len(new_uids) > 0:
                new_uids = np.array(new_uids)
                self.counted_uids[tx_name].update(new_uids)
                self.n_assessed[tx_name] += len(new_uids)

                # Count how many were effective
                effective = treatment.treatment_effective[new_uids]
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

        # Find the HMB cascade intervention
        cascade = None
        for intv in self.sim.interventions.values():
            if hasattr(intv, 'treatments'):
                cascade = intv
                break

        if cascade is None:
            return

        # For each treatment type, check for newly started treatments
        for tx_name in ['nsaid', 'txa', 'pill', 'hiud']:
            treatment = cascade.treatments[tx_name]

            # Get people who accepted this treatment
            accepted = treatment.accepted
            accepted_uids = accepted.uids

            # Find new acceptors we haven't counted yet
            new_uids = [uid for uid in accepted_uids if uid not in self.counted_uids[tx_name]]

            if len(new_uids) > 0:
                # Add to counted set
                self.counted_uids[tx_name].update(new_uids)

                # Get their durations (only for NSAID/TXA, as pill/hIUD use FPsim durations)
                if tx_name in ['nsaid', 'txa']:
                    new_uids_arr = np.array(new_uids)
                    durs = treatment.dur_treatment[new_uids_arr]
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

        # Find the HMB cascade intervention, or return if not found
        cascade = None
        for intv in sim.interventions.values():
            if hasattr(intv, 'treatments'):
                cascade = intv
                break
        if cascade is None:
            return

        # Get the states we need
        menstruating = menstruation.menstruating
        anemic = menstruation.anemic
        pain = menstruation.pain
        hmb = menstruation.hmb
        seeking_care = cascade.treatments['nsaid'].seeking_care  # Use NSAID treatment's seeking_care
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


class track_cascade(ss.Analyzer):
    """
    Track treatment cascade progression and component effects.

    This analyzer monitors:
    1. How many women tried 0, 1, 2, 3, or 4 treatments
    2. Where women drop off along the cascade
    3. Current anemia prevalence by number of treatments tried
    4. Treatment-specific anemia outcomes

    Requires sim.interventions.hmb_care_pathway to be present.

    Example:
        cascade_analyzer = track_cascade()
        sim = fp.Sim(..., analyzers=[cascade_analyzer])
        sim.run()

        # Access results
        print(cascade_analyzer.results.n_tried_1_tx)  # Number who tried exactly 1 treatment
        print(cascade_analyzer.results.anemia_by_cascade)  # Anemia prev by cascade depth
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Track historical data for final attribution analysis
        self.treatment_history = {}  # Will store {uid: {'treatments': [], 'anemia_changes': []}}

    def init_results(self):
        super().init_results()

        # Define results for tracking cascade
        results = [
            # Number of women who have tried each number of treatments (cumulative over time)
            ss.Result('n_tried_0_tx', scale=True, label="Never tried any treatment"),
            ss.Result('n_tried_1_tx', scale=True, label="Tried exactly 1 treatment"),
            ss.Result('n_tried_2_tx', scale=True, label="Tried exactly 2 treatments"),
            ss.Result('n_tried_3_tx', scale=True, label="Tried exactly 3 treatments"),
            ss.Result('n_tried_4_tx', scale=True, label="Tried all 4 treatments"),

            # Proportions (among menstruating women with HMB history)
            ss.Result('prop_tried_0_tx', scale=False, label="Proportion never tried treatment"),
            ss.Result('prop_tried_1_tx', scale=False, label="Proportion tried 1 treatment"),
            ss.Result('prop_tried_2_tx', scale=False, label="Proportion tried 2 treatments"),
            ss.Result('prop_tried_3_tx', scale=False, label="Proportion tried 3 treatments"),
            ss.Result('prop_tried_4_tx', scale=False, label="Proportion tried 4 treatments"),

            # Current treatment status
            ss.Result('on_any_tx', scale=True, label="Currently on any treatment"),
            ss.Result('on_nsaid', scale=True, label="Currently on NSAID"),
            ss.Result('on_txa', scale=True, label="Currently on TXA"),
            ss.Result('on_pill', scale=True, label="Currently on Pill"),
            ss.Result('on_hiud', scale=True, label="Currently on hIUD"),

            # Anemia outcomes by cascade depth (among those who tried N treatments)
            ss.Result('anemia_tried_0', scale=False, label="Anemia prev among those who tried 0 tx"),
            ss.Result('anemia_tried_1', scale=False, label="Anemia prev among those who tried 1 tx"),
            ss.Result('anemia_tried_2', scale=False, label="Anemia prev among those who tried 2 tx"),
            ss.Result('anemia_tried_3', scale=False, label="Anemia prev among those who tried 3 tx"),
            ss.Result('anemia_tried_4', scale=False, label="Anemia prev among those who tried 4 tx"),

            # Treatment-specific anemia outcomes (current and ever)
            ss.Result('anemia_on_nsaid', scale=False, label="Anemia prev among those currently on NSAID"),
            ss.Result('anemia_on_txa', scale=False, label="Anemia prev among those currently on TXA"),
            ss.Result('anemia_on_pill', scale=False, label="Anemia prev among those currently on Pill"),
            ss.Result('anemia_on_hiud', scale=False, label="Anemia prev among those currently on hIUD"),

            ss.Result('anemia_ever_nsaid', scale=False, label="Anemia prev among those who ever tried NSAID"),
            ss.Result('anemia_ever_txa', scale=False, label="Anemia prev among those who ever tried TXA"),
            ss.Result('anemia_ever_pill', scale=False, label="Anemia prev among those who ever tried Pill"),
            ss.Result('anemia_ever_hiud', scale=False, label="Anemia prev among those who ever tried hIUD"),

            # Cascade dropoff tracking (among those who were offered but didn't accept/continue)
            ss.Result('offered_nsaid', scale=True, label="Number offered NSAID"),
            ss.Result('accepted_nsaid', scale=True, label="Number accepted NSAID"),
            ss.Result('offered_txa', scale=True, label="Number offered TXA"),
            ss.Result('accepted_txa', scale=True, label="Number accepted TXA"),
            ss.Result('offered_pill', scale=True, label="Number offered Pill"),
            ss.Result('accepted_pill', scale=True, label="Number accepted Pill"),
            ss.Result('offered_hiud', scale=True, label="Number offered hIUD"),
            ss.Result('accepted_hiud', scale=True, label="Number accepted hIUD"),
        ]

        self.define_results(*results)
        return

    def step(self):
        """Calculate cascade metrics at this timestep"""
        ti = self.sim.ti

        # Get the intervention
        try:
            cascade = self.sim.interventions.hmb_cascade
        except:
            # No intervention present - skip this timestep silently after first warning
            if not hasattr(self, '_warned'):
                print("Warning: track_cascade requires HMBCascade intervention - skipping analysis")
                self._warned = True
            return

        # Get relevant populations
        menstruation = self.sim.people.menstruation
        menstruating = menstruation.menstruating
        anemic = menstruation.anemic

        # Calculate number of treatments tried per person
        n_treatments = (
            np.array(cascade.tried_nsaid, dtype=int) +
            np.array(cascade.tried_txa, dtype=int) +
            np.array(cascade.tried_pill, dtype=int) +
            np.array(cascade.tried_hiud, dtype=int)
        )

        # Count people by number of treatments tried
        tried_0 = (n_treatments == 0) & menstruating
        tried_1 = (n_treatments == 1) & menstruating
        tried_2 = (n_treatments == 2) & menstruating
        tried_3 = (n_treatments == 3) & menstruating
        tried_4 = (n_treatments == 4) & menstruating

        self.results.n_tried_0_tx[ti] = np.count_nonzero(tried_0)
        self.results.n_tried_1_tx[ti] = np.count_nonzero(tried_1)
        self.results.n_tried_2_tx[ti] = np.count_nonzero(tried_2)
        self.results.n_tried_3_tx[ti] = np.count_nonzero(tried_3)
        self.results.n_tried_4_tx[ti] = np.count_nonzero(tried_4)

        # Calculate proportions
        n_menstruating = np.count_nonzero(menstruating)
        if n_menstruating > 0:
            self.results.prop_tried_0_tx[ti] = np.count_nonzero(tried_0) / n_menstruating
            self.results.prop_tried_1_tx[ti] = np.count_nonzero(tried_1) / n_menstruating
            self.results.prop_tried_2_tx[ti] = np.count_nonzero(tried_2) / n_menstruating
            self.results.prop_tried_3_tx[ti] = np.count_nonzero(tried_3) / n_menstruating
            self.results.prop_tried_4_tx[ti] = np.count_nonzero(tried_4) / n_menstruating

        # Current treatment status
        on_nsaid = cascade.treatments['nsaid'].on_treatment & menstruating
        on_txa = cascade.treatments['txa'].on_treatment & menstruating
        on_pill = cascade.treatments['pill'].on_treatment & menstruating
        on_hiud = cascade.treatments['hiud'].on_treatment & menstruating
        on_any = cascade.on_any_treatment & menstruating

        self.results.on_any_tx[ti] = np.count_nonzero(on_any)
        self.results.on_nsaid[ti] = np.count_nonzero(on_nsaid)
        self.results.on_txa[ti] = np.count_nonzero(on_txa)
        self.results.on_pill[ti] = np.count_nonzero(on_pill)
        self.results.on_hiud[ti] = np.count_nonzero(on_hiud)

        # Anemia by cascade depth
        def calc_anemia_prev(mask):
            """Helper to calculate anemia prevalence"""
            n = np.count_nonzero(mask)
            if n > 0:
                return np.count_nonzero(anemic & mask) / n
            return 0

        self.results.anemia_tried_0[ti] = calc_anemia_prev(tried_0)
        self.results.anemia_tried_1[ti] = calc_anemia_prev(tried_1)
        self.results.anemia_tried_2[ti] = calc_anemia_prev(tried_2)
        self.results.anemia_tried_3[ti] = calc_anemia_prev(tried_3)
        self.results.anemia_tried_4[ti] = calc_anemia_prev(tried_4)

        # Anemia among those currently on each treatment
        self.results.anemia_on_nsaid[ti] = calc_anemia_prev(on_nsaid)
        self.results.anemia_on_txa[ti] = calc_anemia_prev(on_txa)
        self.results.anemia_on_pill[ti] = calc_anemia_prev(on_pill)
        self.results.anemia_on_hiud[ti] = calc_anemia_prev(on_hiud)

        # Anemia among those who ever tried each treatment
        ever_nsaid = cascade.tried_nsaid & menstruating
        ever_txa = cascade.tried_txa & menstruating
        ever_pill = cascade.tried_pill & menstruating
        ever_hiud = cascade.tried_hiud & menstruating

        self.results.anemia_ever_nsaid[ti] = calc_anemia_prev(ever_nsaid)
        self.results.anemia_ever_txa[ti] = calc_anemia_prev(ever_txa)
        self.results.anemia_ever_pill[ti] = calc_anemia_prev(ever_pill)
        self.results.anemia_ever_hiud[ti] = calc_anemia_prev(ever_hiud)

        # Track cascade dropoffs (cumulative counts)
        self.results.offered_nsaid[ti] = np.count_nonzero(cascade.treatments['nsaid'].offered)
        self.results.accepted_nsaid[ti] = np.count_nonzero(cascade.treatments['nsaid'].accepted)
        self.results.offered_txa[ti] = np.count_nonzero(cascade.treatments['txa'].offered)
        self.results.accepted_txa[ti] = np.count_nonzero(cascade.treatments['txa'].accepted)
        self.results.offered_pill[ti] = np.count_nonzero(cascade.treatments['pill'].offered)
        self.results.accepted_pill[ti] = np.count_nonzero(cascade.treatments['pill'].accepted)
        self.results.offered_hiud[ti] = np.count_nonzero(cascade.treatments['hiud'].offered)
        self.results.accepted_hiud[ti] = np.count_nonzero(cascade.treatments['hiud'].accepted)

        return

    def finalize(self):
        """Calculate final summary statistics"""
        super().finalize()

        # Only calculate if we have intervention data
        if hasattr(self, '_warned'):
            # No intervention was present, skip finalization
            self.acceptance_rates = None
            self.anemia_reduction = None
            return

        # Calculate acceptance rates for each treatment in the cascade
        def calc_acceptance_rate(offered, accepted):
            """Calculate acceptance rate, handling zero division"""
            if len(offered) > 0 and offered[-1] > 0:
                return accepted[-1] / offered[-1]
            return 0

        self.acceptance_rates = {
            'nsaid': calc_acceptance_rate(self.results.offered_nsaid, self.results.accepted_nsaid),
            'txa': calc_acceptance_rate(self.results.offered_txa, self.results.accepted_txa),
            'pill': calc_acceptance_rate(self.results.offered_pill, self.results.accepted_pill),
            'hiud': calc_acceptance_rate(self.results.offered_hiud, self.results.accepted_hiud),
        }

        # Calculate average anemia prevalence reduction by treatment
        # (comparing those who tried vs those who didn't)
        final_ti = -1  # Last timestep
        self.anemia_reduction = {
            'nsaid': self.results.anemia_tried_0[final_ti] - self.results.anemia_ever_nsaid[final_ti],
            'txa': self.results.anemia_tried_0[final_ti] - self.results.anemia_ever_txa[final_ti],
            'pill': self.results.anemia_tried_0[final_ti] - self.results.anemia_ever_pill[final_ti],
            'hiud': self.results.anemia_tried_0[final_ti] - self.results.anemia_ever_hiud[final_ti],
        }

        return

    def plot_cascade(self, do_save=False, fig_name='cascade_analysis.png'):
        """
        Create visualization of the treatment cascade

        Args:
            do_save: Whether to save the figure
            fig_name: Filename if saving
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Treatment cascade analysis', fontsize=16, y=0.995)

        # Panel 1: Number of treatments tried (final values)
        ax = axes[0, 0]
        treatments_tried = ['0', '1', '2', '3', '4']
        n_tried = [
            self.results.n_tried_0_tx[-1],
            self.results.n_tried_1_tx[-1],
            self.results.n_tried_2_tx[-1],
            self.results.n_tried_3_tx[-1],
            self.results.n_tried_4_tx[-1],
        ]
        ax.bar(treatments_tried, n_tried, color='steelblue', alpha=0.7)
        ax.set_xlabel('Number of treatments tried')
        ax.set_ylabel('Number of women')
        ax.set_title('Distribution of cascade depth')
        ax.grid(axis='y', alpha=0.3)

        # Panel 2: Cascade dropoffs (offer vs accept)
        ax = axes[0, 1]
        treatments = ['NSAID', 'TXA', 'Pill', 'hIUD']
        offered = [
            self.results.offered_nsaid[-1],
            self.results.offered_txa[-1],
            self.results.offered_pill[-1],
            self.results.offered_hiud[-1],
        ]
        accepted = [
            self.results.accepted_nsaid[-1],
            self.results.accepted_txa[-1],
            self.results.accepted_pill[-1],
            self.results.accepted_hiud[-1],
        ]

        x = np.arange(len(treatments))
        width = 0.35
        ax.bar(x - width/2, offered, width, label='Offered', alpha=0.7, color='lightcoral')
        ax.bar(x + width/2, accepted, width, label='Accepted', alpha=0.7, color='seagreen')
        ax.set_xlabel('Treatment')
        ax.set_ylabel('Number of women')
        ax.set_title('Treatment cascade dropoffs')
        ax.set_xticks(x)
        ax.set_xticklabels(treatments)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Panel 3: Anemia prevalence by cascade depth
        ax = axes[1, 0]
        anemia_by_depth = [
            self.results.anemia_tried_0[-1],
            self.results.anemia_tried_1[-1],
            self.results.anemia_tried_2[-1],
            self.results.anemia_tried_3[-1],
            self.results.anemia_tried_4[-1],
        ]
        ax.plot(treatments_tried, anemia_by_depth, marker='o', linewidth=2, markersize=8, color='darkred')
        ax.set_xlabel('Number of treatments tried')
        ax.set_ylabel('Anemia prevalence')
        ax.set_title('Anemia prevalence by cascade depth')
        ax.set_ylim([0, max(anemia_by_depth) * 1.2])
        ax.grid(alpha=0.3)

        # Panel 4: Anemia reduction by treatment
        ax = axes[1, 1]
        reductions = [self.anemia_reduction[t] for t in ['nsaid', 'txa', 'pill', 'hiud']]
        colors = ['green' if r > 0 else 'red' for r in reductions]
        ax.bar(treatments, reductions, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Treatment')
        ax.set_ylabel('Anemia prevalence reduction')
        ax.set_title('Anemia reduction by treatment type')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if do_save:
            plt.savefig(fig_name, dpi=150, bbox_inches='tight')

        return fig


class track_care_propensity_effects(ss.Analyzer):
    """
    Track treatment outcomes by care_seeking_propensity quantiles.

    This analyzer stratifies the population by care_seeking_propensity
    and tracks key outcomes for each quantile:
    - Care-seeking rates
    - Treatment acceptance rates
    - Current treatment status
    - Treatment duration
    - Number of treatments tried

    This allows validation that care_seeking_propensity affects all
    stages of the care cascade as expected.

    Args:
        n_quantiles: Number of quantiles to divide population into (default: 4)
    """

    def __init__(self, n_quantiles=4, **kwargs):
        super().__init__(**kwargs)
        self.n_quantiles = n_quantiles

        # Storage for tracking cumulative metrics
        self.treatment_durations = {i: [] for i in range(n_quantiles)}
        self.n_treatments_tried = {i: [] for i in range(n_quantiles)}

        return

    def init_results(self):
        super().init_results()

        results = []

        # For each quantile, track key metrics
        for i in range(self.n_quantiles):
            results.extend([
                # Care-seeking
                ss.Result(f'care_seeking_q{i}', scale=False,
                         label=f"Care-seeking rate (quantile {i})"),

                # Currently on treatment
                ss.Result(f'on_treatment_q{i}', scale=False,
                         label=f"Proportion on treatment (quantile {i})"),

                # Treatment counts
                ss.Result(f'n_treatments_q{i}', scale=False,
                         label=f"Mean treatments tried (quantile {i})"),
            ])

        self.define_results(*results)
        return

    def step(self):
        """Calculate metrics by quantile at this timestep"""
        ti = self.sim.ti

        # Get the intervention
        try:
            cascade = self.sim.interventions.hmb_cascade
        except:
            if not hasattr(self, '_warned'):
                print("Warning: track_care_propensity_effects requires HMBCascade - skipping")
                self._warned = True
            return

        # Get menstruating women with HMB (eligible population)
        menstruation = self.sim.people.menstruation
        eligible = menstruation.hmb & menstruation.menstruating
        eligible_uids = eligible.uids

        if len(eligible_uids) == 0:
            return

        # Get care_seeking_propensity for eligible population (use NSAID treatment as first-line)
        propensity = cascade.treatments['nsaid'].care_seeking_propensity[eligible_uids]

        # Define quantiles
        quantile_edges = np.percentile(propensity, np.linspace(0, 100, self.n_quantiles + 1))

        # For each quantile, calculate metrics
        for q in range(self.n_quantiles):
            # Get UIDs in this quantile
            if q == 0:
                in_quantile = propensity <= quantile_edges[q + 1]
            elif q == self.n_quantiles - 1:
                in_quantile = propensity > quantile_edges[q]
            else:
                in_quantile = (propensity > quantile_edges[q]) & (propensity <= quantile_edges[q + 1])

            quantile_uids = eligible_uids[in_quantile]

            if len(quantile_uids) == 0:
                continue

            # Calculate care-seeking rate (use NSAID treatment's seeking_care)
            seeking = cascade.treatments['nsaid'].seeking_care[quantile_uids]
            self.results[f'care_seeking_q{q}'][ti] = np.mean(seeking)

            # Calculate proportion currently on treatment
            on_tx = cascade.on_any_treatment[quantile_uids]
            self.results[f'on_treatment_q{q}'][ti] = np.mean(on_tx)

            # Calculate mean number of treatments tried
            n_tx = (
                np.array(cascade.tried_nsaid[quantile_uids], dtype=int) +
                np.array(cascade.tried_txa[quantile_uids], dtype=int) +
                np.array(cascade.tried_pill[quantile_uids], dtype=int) +
                np.array(cascade.tried_hiud[quantile_uids], dtype=int)
            )
            self.results[f'n_treatments_q{q}'][ti] = np.mean(n_tx)

            # Store treatment durations for those currently on treatment
            on_tx_in_q = quantile_uids[on_tx]
            if len(on_tx_in_q) > 0:
                # Calculate how long they've been on current treatment
                # Need to find which treatment each person is on
                time_on_tx = []
                for uid in on_tx_in_q:
                    for treatment in cascade.treatments.values():
                        if treatment.on_treatment[uid]:
                            time_on_tx.append(self.sim.ti - treatment.ti_start_treatment[uid])
                            break
                self.treatment_durations[q].extend(time_on_tx)

        return

    def finalize(self):
        """Calculate final summary statistics"""
        super().finalize()

        # Calculate mean treatment duration by quantile
        self.mean_duration_by_quantile = {}
        for q in range(self.n_quantiles):
            if len(self.treatment_durations[q]) > 0:
                self.mean_duration_by_quantile[q] = np.mean(self.treatment_durations[q])
            else:
                self.mean_duration_by_quantile[q] = 0

        # Calculate final rates (average over last 12 months)
        self.final_care_seeking = {}
        self.final_on_treatment = {}
        self.final_n_treatments = {}

        for q in range(self.n_quantiles):
            # Average over last 12 timesteps (months)
            self.final_care_seeking[q] = np.mean(self.results[f'care_seeking_q{q}'][-12:])
            self.final_on_treatment[q] = np.mean(self.results[f'on_treatment_q{q}'][-12:])
            self.final_n_treatments[q] = np.mean(self.results[f'n_treatments_q{q}'][-12:])

        return


class track_anemia_duration(ss.Analyzer):
    """
    Track cumulative time spent with HMB-related anemia by cascade depth.

    This analyzer tracks the average duration (in months) that women with HMB
    have spent anemic, stratified by how many treatments they've tried.

    The analyzer measures:
    - Mean time with anemia for HMB women at each cascade depth (0-4 treatments)
    - Total person-months of anemia by cascade depth
    - Number of HMB women at each cascade depth

    This helps answer: "Do women who progress through more treatments spend
    less time being anemic overall?"

    Requires:
    - sim.people.menstruation with states: anemic, hmb, menstruating, dur_anemia
    - sim.interventions.hmb_cascade
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def init_results(self):
        """Initialize results storage"""
        super().init_results()

        results = [
            # Average duration with anemia by cascade depth
            ss.Result('mean_dur_anemia_tried_0', scale=False, label="Mean months with anemia (0 treatments)"),
            ss.Result('mean_dur_anemia_tried_1', scale=False, label="Mean months with anemia (1 treatment)"),
            ss.Result('mean_dur_anemia_tried_2', scale=False, label="Mean months with anemia (2 treatments)"),
            ss.Result('mean_dur_anemia_tried_3', scale=False, label="Mean months with anemia (3 treatments)"),
            ss.Result('mean_dur_anemia_tried_4', scale=False, label="Mean months with anemia (4 treatments)"),

            # Total person-months with anemia by cascade depth
            ss.Result('total_dur_anemia_tried_0', scale=True, label="Total person-months anemia (0 treatments)"),
            ss.Result('total_dur_anemia_tried_1', scale=True, label="Total person-months anemia (1 treatment)"),
            ss.Result('total_dur_anemia_tried_2', scale=True, label="Total person-months anemia (2 treatments)"),
            ss.Result('total_dur_anemia_tried_3', scale=True, label="Total person-months anemia (3 treatments)"),
            ss.Result('total_dur_anemia_tried_4', scale=True, label="Total person-months anemia (4 treatments)"),

            # Number of HMB women at each cascade depth
            ss.Result('n_hmb_tried_0', scale=True, label="HMB women (0 treatments)"),
            ss.Result('n_hmb_tried_1', scale=True, label="HMB women (1 treatment)"),
            ss.Result('n_hmb_tried_2', scale=True, label="HMB women (2 treatments)"),
            ss.Result('n_hmb_tried_3', scale=True, label="HMB women (3 treatments)"),
            ss.Result('n_hmb_tried_4', scale=True, label="HMB women (4 treatments)"),
        ]

        self.define_results(*results)
        return

    def step(self):
        """Calculate anemia duration metrics at this timestep"""
        ti = self.sim.ti

        # Get the cascade intervention
        try:
            cascade = self.sim.interventions.hmb_cascade
        except:
            if not hasattr(self, '_warned'):
                print("Warning: track_anemia_duration requires HMBCascade intervention - skipping")
                self._warned = True
            return

        # Get relevant states
        menstruation = self.sim.people.menstruation
        hmb = menstruation.hmb
        menstruating = menstruation.menstruating
        dur_anemia = menstruation.dur_anemia

        # Calculate number of treatments tried per person
        n_treatments = (
            np.array(cascade.tried_nsaid, dtype=int) +
            np.array(cascade.tried_txa, dtype=int) +
            np.array(cascade.tried_pill, dtype=int) +
            np.array(cascade.tried_hiud, dtype=int)
        )

        # For each cascade depth, calculate metrics among HMB women
        for n in range(5):
            # Get HMB women who have tried N treatments
            hmb_tried_n = (n_treatments == n) & hmb & menstruating
            n_hmb = np.count_nonzero(hmb_tried_n)

            # Store number of HMB women at this depth
            self.results[f'n_hmb_tried_{n}'][ti] = n_hmb

            if n_hmb > 0:
                # Get their anemia durations
                durations = dur_anemia[hmb_tried_n]

                # Calculate mean duration
                mean_dur = np.mean(durations)
                self.results[f'mean_dur_anemia_tried_{n}'][ti] = mean_dur

                # Calculate total person-months
                total_dur = np.sum(durations)
                self.results[f'total_dur_anemia_tried_{n}'][ti] = total_dur
            else:
                self.results[f'mean_dur_anemia_tried_{n}'][ti] = 0
                self.results[f'total_dur_anemia_tried_{n}'][ti] = 0

        return

    def finalize(self):
        """Calculate final summary statistics"""
        super().finalize()

        # Calculate final mean durations (average over last 12 months to smooth noise)
        self.final_mean_durations = {}
        for n in range(5):
            self.final_mean_durations[n] = np.mean(self.results[f'mean_dur_anemia_tried_{n}'][-12:])

        return
