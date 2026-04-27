"""
Treatment usage and care-seeking analysis for HMB pool model.

Metrics tracked:
  - Monthly % of HMB women seeking care (by condition: anemia/pain/base)
  - Monthly % on each treatment type
  - Person-months of care-seeking and treatment
  - Distribution of care visits per woman
  - Distribution of treatment duration per woman
  - Gave-up and hysterectomy counts

Uses same 3×3 scenario grid as run_scenarios.py:
  Care:  base (10%), mid (20%), high (35%)
  hIUD:  low (10%), mid (25%), high (40%)
"""

import numpy as np
import sciris as sc
import starsim as ss
import fpsim as fp
import os
import gc
import matplotlib.pyplot as plt
from collections import defaultdict

from menstruation import Menstruation
from education import Education
from interventions_pool import HMBPool, HMBCounterfactual
from analyzers import track_hmb_anemia


# ── Output folders ─────────────────────────────────────────────────────────────
PLOTFOLDER = 'figures_treatment_usage/'
OUTFOLDER  = 'results_treatment_usage/'
for d in [PLOTFOLDER, OUTFOLDER]:
    os.makedirs(d, exist_ok=True)


# ── Settings ───────────────────────────────────────────────────────────────────
P_BASE      = 0.215
P_HMB_PRONE = 0.53
RR_MID      = 1.73
N_SEEDS     = 10
START       = 2020
STOP        = 2030
INTV_YEAR   = 2026


# ── Care-seeking levels (same as run_scenarios.py) ─────────────────────────────

CARE_PRE = sc.objdict(
    p_ever_seek=0.10,
    p_monthly_first=1/36,
    p_monthly_first_anemic=1/12,
    p_monthly_first_pain=1/24,
    p_monthly_repeat=1/6,
)

CARE_LEVELS = {
    'base': sc.objdict(
        label='Base care (10%)',
        care=sc.objdict(
            p_ever_seek=0.10,
            p_monthly_first=1/36,
            p_monthly_first_anemic=1/12,
            p_monthly_first_pain=1/24,
            p_monthly_repeat=1/6,
        ),
    ),
    'mid': sc.objdict(
        label='Mid care (20%)',
        care=sc.objdict(
            p_ever_seek=0.20,
            p_monthly_first=1/12,
            p_monthly_first_anemic=1/6,
            p_monthly_first_pain=1/12,
            p_monthly_repeat=1/6,
        ),
    ),
    'high': sc.objdict(
        label='High care (35%)',
        care=sc.objdict(
            p_ever_seek=0.35,
            p_monthly_first=1/6,
            p_monthly_first_anemic=1/6,
            p_monthly_first_pain=1/6,
            p_monthly_repeat=1/6,
        ),
    ),
}

# ── hIUD uptake levels ─────────────────────────────────────────────────────────
# tx_weights: % of those OFFERED treatment (sums to ~100)

HIUD_LEVELS = {
    'low_hiud': sc.objdict(
        label='Low hIUD (10%)',
        prob_offer=0.73,
        tx_weights=sc.objdict(nsaid=43.2, txa=21.6, pill=21.6, hiud=13.7),
    ),
    'mid_hiud': sc.objdict(
        label='Mid hIUD (25%)',
        prob_offer=0.80,
        tx_weights=sc.objdict(nsaid=34.4, txa=17.2, pill=17.2, hiud=31.3),
    ),
    'high_hiud': sc.objdict(
        label='High hIUD (40%)',
        prob_offer=0.90,
        tx_weights=sc.objdict(nsaid=27.8, txa=13.9, pill=13.9, hiud=44.4),
    ),
}

# ── Colors ─────────────────────────────────────────────────────────────────────
CARE_COLORS = {'base': '#2196F3', 'mid': '#4CAF50', 'high': '#F44336'}
CARE_LABELS = {'base': 'Care 10%', 'mid': 'Care 20%', 'high': 'Care 35%'}
TX_COLORS = {
    'nsaid': '#1f77b4', 'txa': '#ff7f0e',
    'pill': '#2ca02c', 'hiud': '#d62728',
}
TX_LABELS = {
    'nsaid': 'NSAID', 'txa': 'TXA',
    'pill': 'Pill', 'hiud': 'hIUD',
}


# ============================================================================
# Analyzer: detailed treatment usage tracking for pool model
# ============================================================================

class TrackTreatmentUsage(ss.Analyzer):
    """
    Tracks care-seeking and treatment usage statistics for the pool model.

    Works with HMBPool, HMBCounterfactual, or any orchestrator that has
    a .treatments dict and .on_any_treatment property.
    """

    def __init__(self, **kwargs):
        super().__init__(name='track_treatment_usage')
        self._care_visit_counts = None
        self._prev_seeking_uids = None
        self._months_seeking = None
        self._months_on_nsaid = None
        self._months_on_txa = None
        self._months_on_pill = None
        self._months_on_hiud = None
        self._prev_on = None

    def _get_intervention(self):
        for name in ['hmb_pool', 'hmb_counterfactual']:
            if hasattr(self.sim.interventions, name):
                return getattr(self.sim.interventions, name)
        return None

    def init_results(self):
        super().init_results()
        results = []
        for key in [
            'n_hmb_underlying', 'n_menstruating',
            'n_seeking_care', 'n_seeking_care_anemic',
            'n_seeking_care_pain', 'n_seeking_care_base',
            'n_on_nsaid', 'n_on_txa', 'n_on_pill', 'n_on_hiud',
            'n_on_any', 'n_ever_sought',
            'pct_seeking_care', 'pct_on_any',
            'pct_on_nsaid', 'pct_on_txa', 'pct_on_pill', 'pct_on_hiud',
            'pct_seeking_anemic', 'pct_seeking_pain', 'pct_seeking_base',
            'pct_seekers_w_anemia', 'pct_seekers_w_pain', 'pct_seekers_w_base',
            'pm_seeking',
            'pm_on_nsaid', 'pm_on_txa', 'pm_on_pill', 'pm_on_hiud',
            'pm_on_any',
            'n_gave_up', 'n_hysterectomy', 'pct_hysterectomy_all_women',
            'n_ever_seekers',
        ]:
            results.append(ss.Result(key, scale=False, label=key))
        self.define_results(*results)

    def step(self):
        ti = self.ti
        sim = self.sim
        ppl = sim.people
        scale = sim.pars.total_pop / sim.pars.n_agents

        intervention = self._get_intervention()
        if intervention is None:
            return

        mens = ppl.menstruation

        # ── First-call initialization ──
        if self._care_visit_counts is None:
            self._care_visit_counts = defaultdict(int)
            self._prev_seeking_uids = set()
            self._months_seeking = defaultdict(int)
            self._months_on_nsaid = defaultdict(int)
            self._months_on_txa = defaultdict(int)
            self._months_on_pill = defaultdict(int)
            self._months_on_hiud = defaultdict(int)
            self._prev_on = {tx: set() for tx in ['nsaid', 'txa', 'pill', 'hiud']}

        # ── Underlying HMB pool ──
        on_any = intervention.on_any_treatment
        hmb_underlying = (mens.hmb | on_any) & mens.menstruating
        n_hmb = np.count_nonzero(hmb_underlying)
        n_mens = np.count_nonzero(mens.menstruating)

        self.results['n_hmb_underlying'][ti] = n_hmb * scale
        self.results['n_menstruating'][ti] = n_mens * scale

        uids = np.asarray(ppl.uid)

        # ── Care-seeking ──
        seeking = np.zeros(len(ppl), dtype=bool)
        for tx in intervention.treatments.values():
            seeking |= np.asarray(tx.seeking_care)

        current_seeking_uids = set(uids[seeking])
        newly_seeking_uids = current_seeking_uids - self._prev_seeking_uids
        for uid in newly_seeking_uids:
            self._care_visit_counts[uid] += 1
        self._prev_seeking_uids = current_seeking_uids

        for uid in current_seeking_uids:
            self._months_seeking[uid] += 1

        n_seeking = np.count_nonzero(seeking)
        seeking_anemic = seeking & np.asarray(mens.anemic)
        seeking_pain = seeking & np.asarray(mens.pain)
        seeking_base = seeking & ~np.asarray(mens.anemic) & ~np.asarray(mens.pain)

        self.results['n_seeking_care'][ti] = n_seeking * scale
        self.results['n_seeking_care_anemic'][ti] = np.count_nonzero(seeking_anemic) * scale
        self.results['n_seeking_care_pain'][ti] = np.count_nonzero(seeking_pain) * scale
        self.results['n_seeking_care_base'][ti] = np.count_nonzero(seeking_base) * scale
        self.results['pm_seeking'][ti] = n_seeking * scale

        # ── Treatment counts and person-months ──
        tx_month_maps = {
            'nsaid': self._months_on_nsaid,
            'txa':   self._months_on_txa,
            'pill':  self._months_on_pill,
            'hiud':  self._months_on_hiud,
        }

        for tx_name in ['nsaid', 'txa', 'pill', 'hiud']:
            on_tx = np.asarray(intervention.treatments[tx_name].on_treatment)
            on_tx_uids = set(uids[on_tx])
            n_on = len(on_tx_uids)
            self.results[f'n_on_{tx_name}'][ti] = n_on * scale
            self.results[f'pm_on_{tx_name}'][ti] = n_on * scale

            for uid in on_tx_uids:
                tx_month_maps[tx_name][uid] += 1

            self._prev_on[tx_name] = on_tx_uids

        n_on_any_total = np.count_nonzero(on_any)
        n_on_any_hmb = np.count_nonzero(on_any & hmb_underlying)
        self.results['n_on_any'][ti] = n_on_any_total * scale
        self.results['pm_on_any'][ti] = n_on_any_total * scale

        # ── Ever sought care ──
        ever_sought = np.zeros(len(ppl), dtype=bool)
        for tx in intervention.treatments.values():
            ever_sought |= np.asarray(tx.ever_sought_care)
        self.results['n_ever_sought'][ti] = np.count_nonzero(ever_sought & hmb_underlying) * scale

        # ── Ever-seekers (Layer 1) ──
        self.results['n_ever_seekers'][ti] = np.count_nonzero(intervention.ever_seeker) * scale

        # ── Percentages ──
        if n_hmb > 0:
            self.results['pct_seeking_care'][ti] = 100 * n_seeking / n_hmb
            self.results['pct_on_any'][ti] = 100 * n_on_any_hmb / n_hmb
            for tx_name in ['nsaid', 'txa', 'pill', 'hiud']:
                n_on = np.count_nonzero(intervention.treatments[tx_name].on_treatment & hmb_underlying)
                self.results[f'pct_on_{tx_name}'][ti] = 100 * n_on / n_hmb
            self.results['pct_seeking_anemic'][ti] = 100 * np.count_nonzero(seeking_anemic) / n_hmb
            self.results['pct_seeking_pain'][ti] = 100 * np.count_nonzero(seeking_pain) / n_hmb
            self.results['pct_seeking_base'][ti] = 100 * np.count_nonzero(seeking_base) / n_hmb

        if n_seeking > 0:
            self.results['pct_seekers_w_anemia'][ti] = 100 * np.count_nonzero(seeking_anemic) / n_seeking
            self.results['pct_seekers_w_pain'][ti] = 100 * np.count_nonzero(seeking_pain) / n_seeking
            self.results['pct_seekers_w_base'][ti] = 100 * np.count_nonzero(seeking_base) / n_seeking

        # ── Gave up / hysterectomy ──
        self.results['n_gave_up'][ti] = np.count_nonzero(intervention.gave_up) * scale
        self.results['n_hysterectomy'][ti] = np.count_nonzero(intervention.had_hysterectomy) * scale

        n_all_women = np.count_nonzero(ppl.female)
        self.results['pct_hysterectomy_all_women'][ti] = (
            100 * np.count_nonzero(intervention.had_hysterectomy) / max(n_all_women, 1)
        )

    def finalize(self):
        super().finalize()

        intervention = self._get_intervention()
        if intervention is None:
            return

        mens = self.sim.people.menstruation
        hmb_underlying = (mens.hmb | intervention.on_any_treatment) & mens.menstruating
        hmb_uids = np.asarray(hmb_underlying.uids)

        # ── Care visit distribution ──
        if self._care_visit_counts is not None and len(hmb_uids) > 0:
            visits = np.array([self._care_visit_counts.get(uid, 0) for uid in hmb_uids], dtype=int)
            self.results['care_visit_distribution'] = visits
            self.results['care_visit_max'] = int(visits.max())
            self.results['care_visit_mean'] = float(visits.mean())
        else:
            self.results['care_visit_distribution'] = np.array([])
            self.results['care_visit_max'] = 0
            self.results['care_visit_mean'] = 0.0

        # ── Per-person months seeking care ──
        if self._months_seeking is not None and len(hmb_uids) > 0:
            ms = np.array([self._months_seeking.get(uid, 0) for uid in hmb_uids], dtype=int)
            self.results['months_seeking_distribution'] = ms
        else:
            self.results['months_seeking_distribution'] = np.array([])

        # ── Per-person months on each treatment ──
        tx_month_maps = {
            'nsaid': self._months_on_nsaid,
            'txa':   self._months_on_txa,
            'pill':  self._months_on_pill,
            'hiud':  self._months_on_hiud,
        }
        all_uids = np.asarray(self.sim.people.uid)

        for tx_name, month_map in tx_month_maps.items():
            tried = np.asarray(intervention.treatments[tx_name].tried_treatment)
            tried_uids = all_uids[tried]
            if len(tried_uids) > 0:
                dur = np.array([month_map.get(uid, 0) for uid in tried_uids], dtype=int)
                self.results[f'months_on_{tx_name}_distribution'] = dur
            else:
                self.results[f'months_on_{tx_name}_distribution'] = np.array([])

        # ── Combined months on any treatment per HMB woman ──
        if len(hmb_uids) > 0:
            combined = np.zeros(len(hmb_uids), dtype=int)
            for tx_name, month_map in tx_month_maps.items():
                for i, uid in enumerate(hmb_uids):
                    combined[i] += month_map.get(uid, 0)
            self.results['months_on_any_distribution'] = combined
        else:
            self.results['months_on_any_distribution'] = np.array([])

        # ── Episode count from orchestrator ──
        eps = np.asarray(intervention.care_episodes[hmb_uids]) if len(hmb_uids) > 0 else np.array([])
        self.results['care_episodes_distribution'] = eps


# ============================================================================
# Helpers
# ============================================================================

def rr_to_logistic_coeff(rr, p_base=P_BASE):
    p_hmb = np.clip(p_base * rr, 1e-6, 1 - 1e-6)
    return (-np.log(1 / p_hmb - 1)) - (-np.log(1 / p_base - 1))


def make_menstruation():
    coeff = rr_to_logistic_coeff(RR_MID)
    return Menstruation(pars={
        'p_hmb_prone': ss.bernoulli(p=P_HMB_PRONE),
        'hmb_seq': sc.objdict(
            poor_mh=sc.objdict(base=0.4, hmb=1.0),
            anemic=sc.objdict(base=P_BASE, hmb=coeff),
            pain=sc.objdict(base=0.1, hmb=3.36),
        )
    })


def make_intervention(care_name, hiud_name):
    care_info = CARE_LEVELS[care_name]
    hiud_info = HIUD_LEVELS[hiud_name]

    return HMBPool(pars=dict(
        year=2020,
        intv_year=INTV_YEAR,
        care_behavior_pre=CARE_PRE,
        care_behavior_post=care_info.care,
        prob_offer_pre=0.70,
        prob_offer_post=hiud_info.prob_offer,
        tx_weights_pre=sc.objdict(nsaid=50.0, txa=25.0, pill=25.0, hiud=0.0),
        tx_weights_post=hiud_info.tx_weights,
        nsaid=sc.objdict(efficacy=0.33, adherence=0.80),
        txa=sc.objdict(efficacy=0.45, adherence=0.70),
        pill=sc.objdict(efficacy=0.59, adherence=0.80),
        hiud=sc.objdict(efficacy=0.88, adherence=1.00),
    ))


def make_counterfactual():
    return HMBCounterfactual(pars=dict(
        care_behavior_pre=CARE_PRE,
        care_behavior_post=CARE_PRE,
    ))


def make_sim(scenario, seed=0):
    """
    scenario: 'counterfactual' or tuple (care_name, hiud_name)
    """
    mens = make_menstruation()
    edu = Education()

    if scenario == 'counterfactual':
        intervention = make_counterfactual()
    else:
        care_name, hiud_name = scenario
        intervention = make_intervention(care_name, hiud_name)

    return fp.Sim(
        start=START, stop=STOP,
        n_agents=10000, total_pop=55_000_000,
        location='kenya',
        education_module=edu,
        connectors=[mens],
        interventions=[intervention],
        analyzers=[track_hmb_anemia(), TrackTreatmentUsage()],
        rand_seed=seed, verbose=0,
    )


def scenario_key(care_name, hiud_name):
    return f'{care_name}_{hiud_name}'


# ============================================================================
# Run simulations
# ============================================================================

def run_analysis(force_rerun=True):
    results_file = OUTFOLDER + 'treatment_usage_pool_raw.obj'

    if not force_rerun and os.path.exists(results_file):
        print("Loading saved results...")
        return sc.loadobj(results_file)

    # Build scenario list
    all_scenarios = ['counterfactual']
    for care_name in CARE_LEVELS:
        for hiud_name in HIUD_LEVELS:
            all_scenarios.append((care_name, hiud_name))

    raw = {}
    for scen in all_scenarios:
        key = scen if scen == 'counterfactual' else scenario_key(scen[0], scen[1])
        raw[key] = {
            # Monthly time series
            'pct_seeking_care': [], 'pct_seeking_anemic': [],
            'pct_seeking_pain': [], 'pct_seeking_base': [],
            'pct_seekers_w_anemia': [], 'pct_seekers_w_pain': [],
            'pct_seekers_w_base': [],
            'pct_on_any': [], 'pct_on_nsaid': [], 'pct_on_txa': [],
            'pct_on_pill': [], 'pct_on_hiud': [],
            'n_ever_sought': [], 'n_ever_seekers': [],
            'n_gave_up': [], 'n_hysterectomy': [],
            'pct_hysterectomy_all_women': [],
            # Person-month time series
            'pm_seeking': [],
            'pm_on_nsaid': [], 'pm_on_txa': [],
            'pm_on_pill': [], 'pm_on_hiud': [],
            'pm_on_any': [],
            # Snapshot distributions
            'care_visit_distributions': [],
            'care_visit_means': [], 'care_visit_maxes': [],
            'care_episodes_distributions': [],
            'months_seeking_distributions': [],
            'months_on_nsaid_distributions': [],
            'months_on_txa_distributions': [],
            'months_on_pill_distributions': [],
            'months_on_hiud_distributions': [],
            'months_on_any_distributions': [],
            # Anemia
            'hmb_anemia_monthly': [],
        }

    for scen in all_scenarios:
        key = scen if scen == 'counterfactual' else scenario_key(scen[0], scen[1])
        if scen == 'counterfactual':
            label = 'Status quo (10% care, no hIUD)'
        else:
            label = f'{CARE_LEVELS[scen[0]].label} + {HIUD_LEVELS[scen[1]].label}'

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        for seed in range(N_SEEDS):
            print(f"  seed {seed}...", end=" ", flush=True)
            sim = make_sim(scen, seed=seed)
            sim.run()

            tu = sim.results.track_treatment_usage

            # Time series
            for metric in [
                'pct_seeking_care', 'pct_seeking_anemic',
                'pct_seeking_pain', 'pct_seeking_base',
                'pct_on_any', 'pct_on_nsaid', 'pct_on_txa',
                'pct_on_pill', 'pct_on_hiud', 'n_ever_sought',
                'n_ever_seekers', 'n_gave_up', 'n_hysterectomy',
                'pct_hysterectomy_all_women',
                'pct_seekers_w_anemia', 'pct_seekers_w_pain', 'pct_seekers_w_base',
                'pm_seeking', 'pm_on_nsaid', 'pm_on_txa',
                'pm_on_pill', 'pm_on_hiud', 'pm_on_any',
            ]:
                raw[key][metric].append(np.asarray(tu[metric]))

            # Snapshot distributions
            raw[key]['care_visit_distributions'].append(np.asarray(tu['care_visit_distribution']))
            raw[key]['care_visit_means'].append(tu['care_visit_mean'])
            raw[key]['care_visit_maxes'].append(tu['care_visit_max'])
            raw[key]['care_episodes_distributions'].append(np.asarray(tu['care_episodes_distribution']))
            raw[key]['months_seeking_distributions'].append(np.asarray(tu['months_seeking_distribution']))
            for tx_name in ['nsaid', 'txa', 'pill', 'hiud']:
                raw[key][f'months_on_{tx_name}_distributions'].append(
                    np.asarray(tu[f'months_on_{tx_name}_distribution']))
            raw[key]['months_on_any_distributions'].append(np.asarray(tu['months_on_any_distribution']))

            # Anemia
            raw[key]['hmb_anemia_monthly'].append(
                np.asarray(sim.results.track_hmb_anemia['n_anemia_with_hmb']))

            del sim; gc.collect()
            print("done")

    sc.saveobj(results_file, raw)
    print(f"\nSaved: {results_file}")
    return raw


# ============================================================================
# Plotting
# ============================================================================

def plot_care_seeking_breakdown(raw, years_monthly):
    """Stacked area: % of HMB women seeking care by condition. 3 panels by care level."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle('% of HMB women seeking care each month (by condition)\n'
                 f'Mid hIUD (25%), mid RR ({RR_MID})', fontsize=14)
    hiud_name = 'mid_hiud'

    for idx, (care_name, care_info) in enumerate(CARE_LEVELS.items()):
        ax = axes[idx]
        key = scenario_key(care_name, hiud_name)

        pct_base = np.array(raw[key]['pct_seeking_base']).mean(axis=0)
        pct_pain = np.array(raw[key]['pct_seeking_pain']).mean(axis=0)
        pct_anemic = np.array(raw[key]['pct_seeking_anemic']).mean(axis=0)

        ax.fill_between(years_monthly, 0, pct_base,
                        color='#90CAF9', alpha=0.8, label='Base (no anemia/pain)')
        ax.fill_between(years_monthly, pct_base, pct_base + pct_pain,
                        color='#FFB74D', alpha=0.8, label='With pain')
        ax.fill_between(years_monthly, pct_base + pct_pain,
                        pct_base + pct_pain + pct_anemic,
                        color='#EF5350', alpha=0.8, label='With anemia')

        pct_total = np.array(raw[key]['pct_seeking_care']).mean(axis=0)
        ax.plot(years_monthly, pct_total, color='black', lw=1.5, label='Total')

        ax.axvline(INTV_YEAR, color='k', ls='--', lw=1)
        ax.set_xlabel('Year')
        if idx == 0:
            ax.set_ylabel('% of HMB women seeking care')
        ax.set_title(care_info.label)
        ax.set_xlim([START, STOP]); ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc='upper left')

    plt.tight_layout()
    fig.savefig(PLOTFOLDER + 'care_seeking_breakdown.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTFOLDER}care_seeking_breakdown.png")
    return fig


def plot_care_seeker_composition(raw, years_monthly):
    """Among care seekers: % with anemia / pain / neither."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle('Composition of care seekers: % with anemia, pain, or neither\n'
                 f'Mid hIUD (25%), mid RR ({RR_MID})', fontsize=14)
    hiud_name = 'mid_hiud'

    for idx, (care_name, care_info) in enumerate(CARE_LEVELS.items()):
        ax = axes[idx]
        key = scenario_key(care_name, hiud_name)

        pct_base = np.array(raw[key]['pct_seekers_w_base']).mean(axis=0)
        pct_pain = np.array(raw[key]['pct_seekers_w_pain']).mean(axis=0)
        pct_anemia = np.array(raw[key]['pct_seekers_w_anemia']).mean(axis=0)

        ax.fill_between(years_monthly, 0, pct_base,
                        color='#90CAF9', alpha=0.8, label='Neither')
        ax.fill_between(years_monthly, pct_base, pct_base + pct_pain,
                        color='#FFB74D', alpha=0.8, label='With pain')
        ax.fill_between(years_monthly, pct_base + pct_pain,
                        pct_base + pct_pain + pct_anemia,
                        color='#EF5350', alpha=0.8, label='With anemia')

        ax.axvline(INTV_YEAR, color='k', ls='--', lw=1)
        ax.set_xlabel('Year')
        if idx == 0:
            ax.set_ylabel('% of care seekers')
        ax.set_title(care_info.label)
        ax.set_xlim([START, STOP]); ax.set_ylim(0, 100)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc='lower right')

    plt.tight_layout()
    fig.savefig(PLOTFOLDER + 'care_seeker_composition.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTFOLDER}care_seeker_composition.png")
    return fig


def plot_treatment_distribution(raw, years_monthly):
    """Stacked area: % of HMB women on each treatment. 3 panels by care level."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(f'% of HMB women on each treatment\nMid hIUD (25%), mid RR ({RR_MID})', fontsize=14)
    hiud_name = 'mid_hiud'

    for idx, (care_name, care_info) in enumerate(CARE_LEVELS.items()):
        ax = axes[idx]
        key = scenario_key(care_name, hiud_name)

        pct_nsaid = np.array(raw[key]['pct_on_nsaid']).mean(axis=0)
        pct_txa = np.array(raw[key]['pct_on_txa']).mean(axis=0)
        pct_pill = np.array(raw[key]['pct_on_pill']).mean(axis=0)
        pct_hiud = np.array(raw[key]['pct_on_hiud']).mean(axis=0)

        ax.fill_between(years_monthly, 0, pct_nsaid, color=TX_COLORS['nsaid'], alpha=0.8, label='NSAID')
        ax.fill_between(years_monthly, pct_nsaid, pct_nsaid + pct_txa, color=TX_COLORS['txa'], alpha=0.8, label='TXA')
        ax.fill_between(years_monthly, pct_nsaid + pct_txa, pct_nsaid + pct_txa + pct_pill,
                        color=TX_COLORS['pill'], alpha=0.8, label='Pill')
        ax.fill_between(years_monthly, pct_nsaid + pct_txa + pct_pill,
                        pct_nsaid + pct_txa + pct_pill + pct_hiud,
                        color=TX_COLORS['hiud'], alpha=0.8, label='hIUD')

        pct_any = np.array(raw[key]['pct_on_any']).mean(axis=0)
        ax.plot(years_monthly, pct_any, color='black', lw=1.5, label='Any treatment')

        ax.axvline(INTV_YEAR, color='k', ls='--', lw=1)
        ax.set_xlabel('Year')
        if idx == 0:
            ax.set_ylabel('% of HMB women on treatment')
        ax.set_title(care_info.label)
        ax.set_xlim([START, STOP]); ax.set_ylim(0, 30)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc='upper left')

    plt.tight_layout()
    fig.savefig(PLOTFOLDER + 'treatment_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTFOLDER}treatment_distribution.png")
    return fig


def plot_treatment_by_hiud(raw, years_monthly):
    """Treatment distribution by hIUD level. Mid care (20%)."""
    care_name = 'mid'
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(f'Treatment distribution by hIUD uptake\n'
                 f'{CARE_LEVELS[care_name].label}, mid RR ({RR_MID})', fontsize=14)

    for idx, (hiud_name, hiud_info) in enumerate(HIUD_LEVELS.items()):
        ax = axes[idx]
        key = scenario_key(care_name, hiud_name)

        pct_nsaid = np.array(raw[key]['pct_on_nsaid']).mean(axis=0)
        pct_txa = np.array(raw[key]['pct_on_txa']).mean(axis=0)
        pct_pill = np.array(raw[key]['pct_on_pill']).mean(axis=0)
        pct_hiud = np.array(raw[key]['pct_on_hiud']).mean(axis=0)

        ax.fill_between(years_monthly, 0, pct_nsaid, color=TX_COLORS['nsaid'], alpha=0.8, label='NSAID')
        ax.fill_between(years_monthly, pct_nsaid, pct_nsaid + pct_txa, color=TX_COLORS['txa'], alpha=0.8, label='TXA')
        ax.fill_between(years_monthly, pct_nsaid + pct_txa, pct_nsaid + pct_txa + pct_pill,
                        color=TX_COLORS['pill'], alpha=0.8, label='Pill')
        ax.fill_between(years_monthly, pct_nsaid + pct_txa + pct_pill,
                        pct_nsaid + pct_txa + pct_pill + pct_hiud,
                        color=TX_COLORS['hiud'], alpha=0.8, label='hIUD')

        ax.axvline(INTV_YEAR, color='k', ls='--', lw=1)
        ax.set_xlabel('Year')
        if idx == 0:
            ax.set_ylabel('% of HMB women on treatment')
        ax.set_title(hiud_info.label)
        ax.set_xlim([START, STOP]); ax.set_ylim(0, 30)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc='upper left')

    plt.tight_layout()
    fig.savefig(PLOTFOLDER + 'treatment_by_hiud.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTFOLDER}treatment_by_hiud.png")
    return fig


def plot_care_visit_distribution(raw):
    """Histogram of care visits per HMB woman. 3 panels by care level."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig.suptitle(f'Care-seeking visits per HMB woman\nMid hIUD (25%), mid RR ({RR_MID})', fontsize=14)
    hiud_name = 'mid_hiud'

    for idx, (care_name, care_info) in enumerate(CARE_LEVELS.items()):
        ax = axes[idx]
        key = scenario_key(care_name, hiud_name)
        all_visits = np.concatenate(raw[key]['care_visit_distributions'])
        if len(all_visits) == 0:
            continue

        max_visits = int(all_visits.max())
        bins = np.arange(0, min(max_visits + 2, 15)) - 0.5
        ax.hist(all_visits, bins=bins, color=CARE_COLORS[care_name],
                alpha=0.7, edgecolor='black', linewidth=0.5, density=True)

        mean_v = all_visits.mean()
        pct_0 = 100 * np.count_nonzero(all_visits == 0) / len(all_visits)
        pct_3p = 100 * np.count_nonzero(all_visits >= 3) / len(all_visits)

        stats_text = (f'Mean: {mean_v:.1f}\nNever sought: {pct_0:.0f}%\n'
                      f'3+ visits: {pct_3p:.1f}%\nMax: {max_visits}')
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top',
                fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('Number of care-seeking visits')
        if idx == 0:
            ax.set_ylabel('Density')
        ax.set_title(care_info.label)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(PLOTFOLDER + 'care_visit_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTFOLDER}care_visit_distribution.png")
    return fig


def plot_gave_up_hysterectomy(raw, years_monthly):
    """Gave up and hysterectomy counts over time. 3 panels by care level."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(f'Cumulative gave-up and hysterectomy\nMid hIUD (25%), mid RR ({RR_MID})', fontsize=14)
    hiud_name = 'mid_hiud'

    for idx, (care_name, care_info) in enumerate(CARE_LEVELS.items()):
        ax = axes[idx]
        key = scenario_key(care_name, hiud_name)

        gave_up = np.array(raw[key]['n_gave_up']).mean(axis=0)
        hyst = np.array(raw[key]['n_hysterectomy']).mean(axis=0)

        ax.plot(years_monthly, gave_up / 1e6, color='#E57373', lw=2, label='Gave up')
        ax.plot(years_monthly, hyst / 1e6, color='#7B1FA2', lw=2, label='Hysterectomy')

        ax.axvline(INTV_YEAR, color='k', ls='--', lw=1)
        ax.set_xlabel('Year')
        if idx == 0:
            ax.set_ylabel('Cumulative count (millions)')
        ax.set_title(care_info.label)
        ax.set_xlim([START, STOP]); ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
        sc.SIticks(ax=ax)

    plt.tight_layout()
    fig.savefig(PLOTFOLDER + 'gave_up_hysterectomy.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTFOLDER}gave_up_hysterectomy.png")
    return fig


def plot_cumulative_person_months(raw, years_monthly):
    """Cumulative person-months on each treatment. 3 panels by care level."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    fig.suptitle(f'Cumulative person-months of treatment\nMid hIUD (25%), mid RR ({RR_MID})', fontsize=14)
    hiud_name = 'mid_hiud'

    for idx, (care_name, care_info) in enumerate(CARE_LEVELS.items()):
        ax = axes[idx]
        key = scenario_key(care_name, hiud_name)

        cum_seek = np.cumsum(np.array(raw[key]['pm_seeking']).mean(axis=0)) / 1e6
        cum_nsaid = np.cumsum(np.array(raw[key]['pm_on_nsaid']).mean(axis=0)) / 1e6
        cum_txa = np.cumsum(np.array(raw[key]['pm_on_txa']).mean(axis=0)) / 1e6
        cum_pill = np.cumsum(np.array(raw[key]['pm_on_pill']).mean(axis=0)) / 1e6
        cum_hiud = np.cumsum(np.array(raw[key]['pm_on_hiud']).mean(axis=0)) / 1e6

        ax.plot(years_monthly, cum_seek, color='grey', lw=2, ls='--', label='Seeking care')
        ax.fill_between(years_monthly, 0, cum_nsaid, color=TX_COLORS['nsaid'], alpha=0.7, label='NSAID')
        ax.fill_between(years_monthly, cum_nsaid, cum_nsaid + cum_txa, color=TX_COLORS['txa'], alpha=0.7, label='TXA')
        ax.fill_between(years_monthly, cum_nsaid + cum_txa, cum_nsaid + cum_txa + cum_pill,
                        color=TX_COLORS['pill'], alpha=0.7, label='Pill')
        ax.fill_between(years_monthly, cum_nsaid + cum_txa + cum_pill,
                        cum_nsaid + cum_txa + cum_pill + cum_hiud,
                        color=TX_COLORS['hiud'], alpha=0.7, label='hIUD')

        ax.axvline(INTV_YEAR, color='k', ls='--', lw=1)
        ax.set_xlabel('Year')
        if idx == 0:
            ax.set_ylabel('Cumulative person-months (millions)')
        ax.set_title(care_info.label)
        ax.set_xlim([START, STOP]); ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc='upper left')

    plt.tight_layout()
    fig.savefig(PLOTFOLDER + 'cumulative_person_months.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTFOLDER}cumulative_person_months.png")
    return fig


def plot_treatment_duration_distributions(raw):
    """Months on treatment per woman who tried it. 3 rows × 4 columns."""
    hiud_name = 'mid_hiud'
    care_list = list(CARE_LEVELS.keys())
    tx_list = ['nsaid', 'txa', 'pill', 'hiud']

    fig, axes = plt.subplots(len(care_list), len(tx_list), figsize=(20, 12), sharey='row')
    fig.suptitle(f'Months on treatment per woman who tried it\n'
                 f'Mid hIUD (25%), mid RR ({RR_MID})', fontsize=14)

    for row, care_name in enumerate(care_list):
        key = scenario_key(care_name, hiud_name)
        for col, tx_name in enumerate(tx_list):
            ax = axes[row, col]
            all_dur = np.concatenate(raw[key][f'months_on_{tx_name}_distributions'])

            if len(all_dur) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                continue

            used = all_dur[all_dur > 0]
            if len(used) == 0:
                ax.text(0.5, 0.5, 'No usage', ha='center', va='center',
                        transform=ax.transAxes, fontsize=12)
                continue

            max_dur = int(used.max())
            bins = np.arange(0.5, min(max_dur + 1.5, 50), 1)
            ax.hist(used, bins=bins, color=TX_COLORS[tx_name],
                    alpha=0.7, edgecolor='black', linewidth=0.5, density=True)

            stats_text = (f'Mean: {used.mean():.1f} mo\n'
                          f'Median: {np.median(used):.0f} mo\n'
                          f'N tried: {len(all_dur)}')
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                    ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlabel('Months on treatment')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            if row == 0:
                ax.set_title(TX_LABELS[tx_name], fontsize=12)
            if col == 0:
                ax.set_ylabel(f'{CARE_LEVELS[care_name].label}\nDensity')

    plt.tight_layout()
    fig.savefig(PLOTFOLDER + 'treatment_duration_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTFOLDER}treatment_duration_distributions.png")
    return fig


# ============================================================================
# Summary tables
# ============================================================================

def print_treatment_summary(raw):
    hiud_name = 'mid_hiud'
    print(f"\n{'═'*100}")
    print(f"  TREATMENT USAGE SUMMARY (end-of-sim, mid hIUD 25%)")
    print(f"{'═'*100}")

    print(f"\n  {'Scenario':<20}  {'On any':>8}  {'NSAID':>8}  {'TXA':>8}  "
          f"{'Pill':>8}  {'hIUD':>8}  {'Seeking':>8}")
    print(f"  {'─'*80}")

    for key_label, key_val in [('Status quo', 'counterfactual')] + \
            [(CARE_LEVELS[c].label, scenario_key(c, hiud_name)) for c in CARE_LEVELS]:
        pct_any = np.array(raw[key_val]['pct_on_any'])[:, -1].mean()
        pct_nsaid = np.array(raw[key_val]['pct_on_nsaid'])[:, -1].mean()
        pct_txa = np.array(raw[key_val]['pct_on_txa'])[:, -1].mean()
        pct_pill = np.array(raw[key_val]['pct_on_pill'])[:, -1].mean()
        pct_hiud = np.array(raw[key_val]['pct_on_hiud'])[:, -1].mean()
        pct_seek = np.array(raw[key_val]['pct_seeking_care'])[:, -1].mean()
        print(f"  {key_label:<20}  {pct_any:>7.1f}%  {pct_nsaid:>7.1f}%  "
              f"{pct_txa:>7.1f}%  {pct_pill:>7.1f}%  {pct_hiud:>7.1f}%  {pct_seek:>7.1f}%")

    # Full 3×3 grid
    print(f"\n  FULL 3×3: % of HMB women on any treatment (end-of-sim)")
    print(f"  {'':>20}", end="")
    for hiud_name, hiud_info in HIUD_LEVELS.items():
        print(f"  {hiud_info.label:>22}", end="")
    print()
    print(f"  {'─'*90}")

    for care_name, care_info in CARE_LEVELS.items():
        print(f"  {care_info.label:<20}", end="")
        for hiud_name in HIUD_LEVELS:
            key = scenario_key(care_name, hiud_name)
            pct = np.array(raw[key]['pct_on_any'])[:, -1]
            print(f"  {pct.mean():>7.1f}±{pct.std():.1f}%", end="")
        print()

    print(f"\n{'═'*100}")


def print_gave_up_summary(raw):
    hiud_name = 'mid_hiud'
    print(f"\n{'═'*80}")
    print(f"  GAVE UP / HYSTERECTOMY (end-of-sim, mid hIUD 25%)")
    print(f"{'═'*80}")

    print(f"\n  {'Scenario':<20}  {'Gave up (M)':>14}  {'Hysterectomy (M)':>18}  {'Hyst % women':>14}")
    print(f"  {'─'*70}")

    for key_label, key_val in [('Status quo', 'counterfactual')] + \
            [(CARE_LEVELS[c].label, scenario_key(c, hiud_name)) for c in CARE_LEVELS]:
        gu = np.array(raw[key_val]['n_gave_up'])[:, -1].mean() / 1e6
        hy = np.array(raw[key_val]['n_hysterectomy'])[:, -1].mean() / 1e6
        pct_h = np.array(raw[key_val]['pct_hysterectomy_all_women'])[:, -1].mean()
        print(f"  {key_label:<20}  {gu:>10.2f}M  {hy:>14.2f}M  {pct_h:>12.2f}%")

    print(f"\n{'═'*80}")


def print_person_month_summary(raw):
    hiud_name = 'mid_hiud'
    print(f"\n{'═'*100}")
    print(f"  PERSON-MONTH SUMMARY (mid hIUD 25%)")
    print(f"{'═'*100}")

    print(f"\n  Cumulative person-months (millions, full sim {START}–{STOP}):")
    print(f"  {'Scenario':<20}  {'Seeking':>10}  {'NSAID':>10}  {'TXA':>10}  "
          f"{'Pill':>10}  {'hIUD':>10}  {'Any Tx':>10}")
    print(f"  {'─'*85}")

    for key_label, key_val in [('Status quo', 'counterfactual')] + \
            [(CARE_LEVELS[c].label, scenario_key(c, hiud_name)) for c in CARE_LEVELS]:
        cum_seek = np.array(raw[key_val]['pm_seeking']).sum(axis=1).mean() / 1e6
        cum_nsaid = np.array(raw[key_val]['pm_on_nsaid']).sum(axis=1).mean() / 1e6
        cum_txa = np.array(raw[key_val]['pm_on_txa']).sum(axis=1).mean() / 1e6
        cum_pill = np.array(raw[key_val]['pm_on_pill']).sum(axis=1).mean() / 1e6
        cum_hiud = np.array(raw[key_val]['pm_on_hiud']).sum(axis=1).mean() / 1e6
        cum_any = np.array(raw[key_val]['pm_on_any']).sum(axis=1).mean() / 1e6
        print(f"  {key_label:<20}  {cum_seek:>9.1f}M  {cum_nsaid:>9.1f}M  "
              f"{cum_txa:>9.1f}M  {cum_pill:>9.1f}M  {cum_hiud:>9.1f}M  {cum_any:>9.1f}M")

    # Per-person treatment duration
    print(f"\n  Mean months on treatment per woman who tried it:")
    print(f"  {'Scenario':<20}  {'NSAID':>10}  {'TXA':>10}  {'Pill':>10}  {'hIUD':>10}")
    print(f"  {'─'*65}")

    for key_label, key_val in [('Status quo', 'counterfactual')] + \
            [(CARE_LEVELS[c].label, scenario_key(c, hiud_name)) for c in CARE_LEVELS]:
        means = []
        for tx_name in ['nsaid', 'txa', 'pill', 'hiud']:
            all_dur = np.concatenate(raw[key_val][f'months_on_{tx_name}_distributions'])
            used = all_dur[all_dur > 0] if len(all_dur) > 0 else np.array([0])
            means.append(used.mean() if len(used) > 0 else 0)
        print(f"  {key_label:<20}  {means[0]:>9.1f}m  {means[1]:>9.1f}m  "
              f"{means[2]:>9.1f}m  {means[3]:>9.1f}m")

    print(f"\n{'═'*100}")


def print_care_visit_summary(raw):
    hiud_name = 'mid_hiud'
    print(f"\n{'═'*80}")
    print(f"  CARE VISIT DISTRIBUTION (mid hIUD 25%)")
    print(f"{'═'*80}")

    print(f"\n  {'Scenario':<20}  {'Mean':>6}  {'Median':>8}  {'Max':>6}  "
          f"{'Never':>8}  {'3+':>8}")
    print(f"  {'─'*65}")

    for key_label, key_val in [('Status quo', 'counterfactual')] + \
            [(CARE_LEVELS[c].label, scenario_key(c, hiud_name)) for c in CARE_LEVELS]:
        all_visits = np.concatenate(raw[key_val]['care_visit_distributions'])
        if len(all_visits) == 0:
            continue
        pct_0 = 100 * np.count_nonzero(all_visits == 0) / len(all_visits)
        pct_3p = 100 * np.count_nonzero(all_visits >= 3) / len(all_visits)
        print(f"  {key_label:<20}  {all_visits.mean():>5.1f}  {np.median(all_visits):>7.0f}  "
              f"{all_visits.max():>5.0f}  {pct_0:>7.1f}%  {pct_3p:>7.1f}%")

    print(f"\n{'═'*80}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':

    do_run = True
    raw = run_analysis(force_rerun=do_run)

    # Time axes
    first_key = list(raw.keys())[0]
    n_months = len(raw[first_key]['pct_seeking_care'][0])
    years_monthly = np.array([START + m / 12 for m in range(n_months)])

    # Plots
    plot_care_seeking_breakdown(raw, years_monthly)
    plot_care_seeker_composition(raw, years_monthly)
    plot_treatment_distribution(raw, years_monthly)
    plot_treatment_by_hiud(raw, years_monthly)
    plot_care_visit_distribution(raw)
    plot_gave_up_hysterectomy(raw, years_monthly)
    plot_cumulative_person_months(raw, years_monthly)
    plot_treatment_duration_distributions(raw)

    # Summary tables
    print_treatment_summary(raw)
    print_gave_up_summary(raw)
    print_person_month_summary(raw)
    print_care_visit_summary(raw)