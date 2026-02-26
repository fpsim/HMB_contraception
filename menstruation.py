"""
Heavy Menstrual Bleeding
Adds an agent state to proxy heavy menstrual bleeding and initializes state
"""
import numpy as np
import sciris as sc
import starsim as ss
from utils import logistic



# To do: generalize code to allow for sensitivity analysis over effectiveness parameters
"""
To calculate the intercept values from the interventions' odds of hmb:
# define desired odds of hmb, given intervention
odds_hmb_base = 0.5, # assumption
odds_hmb_hiud =  (1-0.312) * 0.5, # Ref (Park, 2015)
odds_hmb_txa = (1 - 0.5*0.312) * 0.5, # 50% effectiveness of hiud, Ref (Bofill Rodriguez, 2022)
odds_hmb_pill = (1 - 0.25*0.312) * 0.5, # 25% effectiveness of hiud, Ref (Bofill Rodriguez, 2022)
# calculate par value which is fed into logistic regression
# Odds with intervention: 1/(1+exp(-(0+intercept)))=odds_intervention
# i.e. to get the intercept from the odds, we calculate: -np.log(1/odds_intervention -1) - intercept_base
intercept_base = np.log(1/odds_hmb_base -1),
intercept_hiud = -np.log(1/odds_hmb_hiud -1) - intercept_base,
intercept_txa = -np.log(1/odds_hmb_txa -1) - intercept_base,
intercept_pill = -np.log(1/odds_hmb_pill -1) - intercept_base,
"""


class Menstruation(ss.Connector):
    """
    Class to handle menstruation related events
    """
    
    def __init__(self, pars=None, name='menstruation', **kwargs):
        super().__init__(name=name)

        # Define parameters
        self.define_pars(
            unit='month',
            
            # --- Menses
            age_menses=ss.lognorm_ex(14, 3),  # Age of menarche
            age_menopause=ss.normal(50, 3),  # Age of menopause
            eff_hyst_menopause=ss.normal(-5, 1),  # Adjustment for age of menopause if hysterectomy occurs

            # HMB prediction
            p_hmb_prone=ss.bernoulli(p=0.486),  # Proportion of menstruating women who experience HMB (sans interventions)
            
            # Odds ratios to create an age curve (from UW Start) ---
            hmb_age_OR = {
            "10-19": 1.00, # prevalence of 0.3
            "20-29": 0.58, # prevalence of 0.2
            "30-39": 0.58, # prevalence of 0.2
            "40-49": 1.00, # prevalence of 0.3
            },

            hmb_pred=sc.objdict(  # Parameters for HMB prediction
                # Baseline probability that those prone to HMB will experience it at any point in time.
                # This is converted to an intercept in the logistic regression: -np.log(1/base-1)
                # Treatment effects are now handled via responder status in HMBCarePathway intervention
                base=0.995,
            ),

            # Non-permanent sequelae of HMB
            hmb_seq=sc.objdict(
                poor_mh=sc.objdict(  # Parameters for poor menstrual hygiene
                    base = 0.4,  # Intercept for poor menstrual hygiene
                    hmb = 1,  # Effect of HMB on poor menstrual hygiene - placeholder
                ),
                anemic=sc.objdict(  # Parameters for anemia
                    # This is converted to an intercept in the logistic regression: -np.log(1/base-1)
                    base = 0.215,  # Baseline probability of anemia; UW Start used the prevalence of anemia in general population from GBD 2021 to estimate RR of 1.73 from OR of 2.17 which gives the baseline risk of 21.5%. 
                    hmb = -np.log(1/0.372 - 1) + np.log(1/0.215 - 1), 
                ),
                pain=sc.objdict(  # Parameters for menstrual pain
                    base = 0.1,  # Baseline probability of menstrual pain
                    hmb = 1.5,  # Effect of HMB on menstrual pain - placeholder: prob of pain is 1/(1+np.exp(-(-np.log(1/0.1 -1)+1.5))) = 0.332
                ),
            ),

            # Permanent sequelae of HMB
            hyst=sc.objdict(  # Parameters for hysterectomy
                base=0.01,  # Baseline probability of hysterectomy
                hmb=2,      # Effect of HMB on hysterectomy - placeholder
                lt40=-5,    # Adjustment for women less than 40
            ),
        )
        
        self.update_pars(pars, **kwargs)

        # Probabilities of various outcomes - all set via models within the module
        # Don't directly alter anything here, as the probabilities are calculated
        # via logistic regression models based on the parameters defined in the main
        # parameter dict.
        self._p_hmb = ss.bernoulli(p=0)
        self._p_poor_mh = ss.bernoulli(p=0)
        self._p_anemic = ss.bernoulli(p=0)
        self._p_pain = ss.bernoulli(p=0)
        self._p_hyst = ss.bernoulli(p=0)

        # Define states
        self.define_states(
            # HMB states
            ss.BoolState('hmb_prone'),
            ss.BoolState('hmb'),
            ss.BoolState('hmb_sus', label="Susceptible to HMB"),

            # HMB sequelae
            ss.BoolState('anemic'),
            ss.FloatArr('dur_anemia', default=0),
            ss.BoolState('prev_anemic', default=False),
            ss.BoolState('poor_mh', label="Poor menstrual hygiene"),
            ss.BoolState('pain', label="Menstrual pain"),
            ss.BoolState('hyst', label="Hysterectomy"),

            # Menstrual states
            ss.BoolState('menstruating'),
            ss.BoolState('premenarchal'),
            ss.BoolState('post_menarche'),
            ss.BoolState('menopausal'),
            ss.BoolState('early_meno'),
            ss.BoolState('premature_meno'),
            ss.FloatArr('age_menses', label="Age of menarche"),
            ss.FloatArr('age_menopause', label="Age of menopause"),
        )

        return

    def init_results(self):
        """ Initialize results """
        super().init_results()
        results = [
            ss.Result('hmb_prev', scale=False, label="Prevalence of HMB"),
            ss.Result('poor_mh_prev', scale=False, label="Prevalence of poor menstrual hygiene"),
            ss.Result('anemic_prev', scale=False, label="Prevalence of anemia"),
            ss.Result('pain_prev', scale=False, label="Prevalence of menstrual pain"),
            ss.Result('hyst_prev', scale=False, label="Prevalence of hysterectomy"),
            ss.Result('early_meno_prev', scale=False, label="Early menopause prevalence"),
            ss.Result('premature_meno_prev', scale=False, label="Premature menopause prevalence"),
            ss.Result('n_anemia', scale=True, label="Cumulative anemia cases"),
        ]
        self.define_results(*results)
        return
    
    # property: less than 40 years old
    @property
    def lt40(self):
        return (self.sim.people.age < 40) & self.sim.people.female
    
    # get ids of women within age limit ( < upper_age )
    def _get_uids(self, upper_age=None):
        """ Get uids of females younger than upper_age """
        people = self.sim.people
        if upper_age is None: upper_age = 1000
        within_age = people.age < float(upper_age)
        
        return (within_age & people.female).uids

    def set_mens_states(self, upper_age=None):
        """ Set menstrual states """
        f_uids = self._get_uids(upper_age=upper_age)
        self.age_menses[f_uids] = self.pars.age_menses.rvs(f_uids)
        self.age_menopause[f_uids] = self.pars.age_menopause.rvs(f_uids)
        self.hmb_prone[f_uids] = self.pars.p_hmb_prone.rvs(f_uids)
        
        return

    def init_post(self):
        """ Initialize with sim properties """
        super().init_post()

        # Set initial menstrual states
        self.set_mens_states()
        self.set_hmb(self.hmb_sus.uids)
        
        # Initialize tracking for cumulative anemia cases
        self._annual_anemia_cases = 0  
        self._last_year_anemia = self.sim.t.year  

        return

    def set_hmb(self, uids):
        """
        Set who will experience heavy menstrual bleeding (HMB).

        HMB is determined by:
        1. Base probability (for hmb_prone individuals)
        2. Age-specific odds ratios

        Note: Treatment effects are applied by HMBCarePathway intervention,
        which removes HMB from treatment responders.
        """
        # Calculate the probability of HMB based on base probability and age
        # Note: Despite comment on line 58, the parameter is treated as probability here
        intercept = -np.log(1/self.pars.hmb_pred.base-1)
        rhs = np.full_like(uids, fill_value=intercept, dtype=float)

        # Apply age-specific odds ratios
        ages = self.sim.people.age[uids]
        age_adjustments = np.zeros_like(uids, dtype=float)
        # Create age masks
        age_10_19 = (ages >= 15) & (ages < 20)
        age_20_29 = (ages >= 20) & (ages < 30)
        age_30_39 = (ages >= 30) & (ages < 40)
        age_40_plus = ages >= 40
        # Apply log(OR) for each age group
        age_adjustments[age_10_19] = np.log(self.pars.hmb_age_OR["10-19"])
        age_adjustments[age_20_29] = np.log(self.pars.hmb_age_OR["20-29"])
        age_adjustments[age_30_39] = np.log(self.pars.hmb_age_OR["30-39"])
        age_adjustments[age_40_plus] = np.log(self.pars.hmb_age_OR["40-49"])
        rhs += age_adjustments

        # Calculate the probability
        p_hmb = 1 / (1+np.exp(-rhs))

        self._p_hmb.set(0)
        self._p_hmb.set(p_hmb)
        # filter draws Bernoulli random variables to determine who gets HMB
        has_hmb = self._p_hmb.filter(uids)
        # set hmb among those uids filtered above
        self.hmb[has_hmb] = True

        return

    def _set_hmb_sequelae(self):
        """
        Set non-permanent sequelae of HMB (anemia, pain, poor menstrual hygiene).

        Called from finish_step() to ensure sequelae reflect post-intervention HMB status.
        """
        mens_uids = self.menstruating.uids

        # Set non-permanent sequelae of HMB
        for seq, p in self.pars.hmb_seq.items():
            old_attr = getattr(self, seq)
            old_attr[:] = False  # Reset the state
            self.setattribute(seq, old_attr)
            attr_dist = getattr(self, f'_p_{seq}')
            attr_dist.set(0)

            # Calculate the probability of the sequelae
            p_val = logistic(self, mens_uids, p)
            attr_dist = getattr(self, f'_p_{seq}')
            attr_dist.set(p_val)
            has_attr = attr_dist.filter(mens_uids)
            new_attr = getattr(self, seq)
            new_attr[has_attr] = True
            self.setattribute(seq, new_attr)

        return

    def step(self):
        """ Updates for this timestep """

        # Set menstruating states
        self.set_mens_states(upper_age=self.t.dt)  # for new agents
        self.step_states()  # for existing agents

        mens_uids = self.menstruating.uids
        self.hmb[:] = False  # Reset

        # Recalculate hmb_sus to ensure it reflects current state before setting HMB
        # (other modules may have updated pregnancy status since step_states was called)
        self.hmb_sus[:] = self.menstruating & self.hmb_prone & ~self.sim.people.fp.pregnant

        # Update HMB
        self.set_hmb(self.hmb_sus.uids)

        # Note: HMB sequelae (anemia, pain, poor_mh) are calculated in finish_step()
        # to ensure they reflect post-intervention HMB status

        # Set hysterectomy state (permanent sequela, kept in step())
        hyst_sus = (self.menstruating & ~self.hyst).uids
        p_hyst = logistic(self, hyst_sus, self.pars.hyst)
        self._p_hyst.set(0)
        self._p_hyst.set(p_hyst)
        has_hyst = self._p_hyst.filter(hyst_sus)
        self.hyst[has_hyst] = True

        # For women who've had a hysterectomy, reset age of menopause
        eff_hyst_menopause = self.pars.eff_hyst_menopause.rvs(has_hyst)
        self.age_menopause[has_hyst] += eff_hyst_menopause

        return

    def step_states(self):
        """ Updates for this timestep """
        ppl = self.sim.people
        cm = self.sim.connectors.contraception
        f = ppl.female

        self.premenarchal[:] = f & (ppl.age < self.age_menses)
        self.post_menarche[:] = f & (ppl.age > self.age_menses)
        self.menstruating[:] = f & (ppl.age <= self.age_menopause) & (ppl.age >= self.age_menses)
        self.menopausal[:] = f & (ppl.age > self.age_menopause)
        self.early_meno[:] = self.menopausal & (self.age_menopause < 45)
        self.premature_meno[:] = self.menopausal & (self.age_menopause < 40)
        self.hmb_sus[:] = self.menstruating & self.hmb_prone & ~self.sim.people.fp.pregnant

        return

    def finish_step(self):
        """
        Finish the timestep after interventions have run.

        Calculate HMB sequelae here to ensure they reflect post-intervention HMB status.
        This is called after interventions.step() but before results are finalized.
        """
        # Calculate HMB sequelae after interventions have modified HMB status
        self._set_hmb_sequelae()

        super().finish_step()
        return

    def finalize(self):
        """
        Finalize state after all modules have updated.

        This ensures that any state changes made by other modules (e.g., pregnancy,
        interventions) after the menstruation module's step() are properly reflected
        in HMB state and its sequelae.
        """
        # Clear HMB for any women who are currently pregnant
        # (handles case where pregnancy occurred after menstruation module ran)
        if hasattr(self.sim.people, 'fp'):
            pregnant = self.sim.people.fp.pregnant
            self.hmb[pregnant] = False
            self.hmb_sus[pregnant] = False

        super().finalize()

    def update_results(self):
        super().update_results()
        ti = self.ti
        def count(arr): return np.count_nonzero(arr)
        def cond_prob(a, b): return sc.safedivide(count(a & b), count(b))
        for res in ['hmb', 'poor_mh', 'anemic', 'pain', 'hyst', 'early_meno', 'premature_meno']:
            self.results[f'{res}_prev'][ti] = cond_prob(getattr(self, res), self.menstruating)
        for res in ['hyst', 'early_meno', 'premature_meno']:
            self.results[f'{res}_prev'][ti] = cond_prob(getattr(self, res), self.post_menarche)

        # --- cumulative anemia cases per year ---
        mens = self.menstruating
        new_cases = (~self.prev_anemic) & self.anemic & mens
        current_new_cases = np.count_nonzero(new_cases)

        current_year = self.sim.t.year
        if current_year != self._last_year_anemia:
            self._annual_anemia_cases = current_new_cases
            self._last_year_anemia = current_year
        else:
            self._annual_anemia_cases += current_new_cases

        self.results.n_anemia[ti] = self._annual_anemia_cases
        self.prev_anemic[:] = self.anemic[:]

        # --- increment anemia duration (in months) ---
        # Increment duration for all currently anemic people
        anemic_uids = self.anemic.uids
        self.dur_anemia[anemic_uids] += 1

        # Reset duration for people who recovered from anemia
        recovered = (~self.anemic) & (self.dur_anemia > 0)
        self.dur_anemia[recovered] = 0

        return



