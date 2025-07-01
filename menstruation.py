"""
Heavy Menstrual Bleeding
Adds an agent state to proxy heavy menstrual bleeding and initializes state 
"""
import fpsim as fp
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss


class Menstruation(ss.Connector):
    
    '''Create a class to handle menstruation related events'''
    
    def __init__(self, pars=None, name='menstruation', **kwargs):
        super().__init__(name=name)

        # Define parameters
        self.define_pars(
            unit='month',
            
            # Menses
            age_menses=ss.lognorm_ex(14, 3),  # Age of menarche
            age_menopause=ss.normal(50, 3),  # Age of menopause ##TODO: Allow for early menopause, generate a flag

            # IUD usage
            p_hiud=ss.bernoulli(p=0.05),  # Baseline, can be modified by interventions

            # HMB prediction
            p_hmb_prone=ss.bernoulli(p=0.4),  # Proportion of menstruating women who experience HMB (sans interventions)
            hmb_pred=sc.objdict(  # Parameters for HMB prediction
                base=0.95,  # For those prone to HMB, probability they'll experience it this timestep
                hiud=-10,  # Effect of IUD on HMB - placeholder
            ),

            # Non-permanent sequelae of HMB
            hmb_seq=sc.objdict(
                poor_mh=sc.objdict(  # Parameters for poor menstrual hygiene
                    base=0.4,  # Intercept for poor menstrual hygiene
                    hiud=-0.5,  # Effect of IUD on poor menstrual hygiene - placeholder ##TODO: Allow for an effect of other hormonal contraception
                ),
                anemic=sc.objdict(  # Parameters for anemia
                    base=0.01,  # Baseline probability of anemia
                    hmb=1.5,  # Effect of HMB on anemia - placeholder
                ),
                pain=sc.objdict(  # Parameters for menstrual pain
                    base=0.1,  # Baseline probability of menstrual pain
                    hmb=1.5,  # Effect of HMB on menstrual pain - placeholder
                    hiud=-0.5,  # Effect of IUD on menstrual pain - placeholder ##TODO: Other contraceptive methods
                ),
                # education=sc.objdict(
                #     base=0.5,  # probability that HMB causes education disruption
                # ),
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
            ss.State('hmb_prone'),  # Prone to HMB
            ss.State('hmb'),
            ss.State('anemic'),
            ss.State('poor_mh', label="Poor menstrual hygiene"),
            ss.State('pain', label="Menstrual pain"),
            ss.State('hyst', label="Hysterectomy"),
            ss.State('hiud', label="Hormonal IUD usage"),
            ss.State('menopausal', label='Has entered menopause'),
            ss.State('early_meno', label='Early menopause'),
            ss.State('premature_meno', label='Premature menopause'),
            ss.FloatArr('age_at_menopause', label='Age of menopause onset'),
            ss.FloatArr('age_menses', label="Age of menarche"),
            ss.FloatArr('age_menopause', label="Age of menopause"),
        )

        return

    def set_early_menopause(self):
        """
        NOT FUNCTIONAL YET
        Set menopause status based on age or hysterectomy.
        - Women enter menopause naturally at age >= age_menopause.
        - Early menopause occurs if hysterectomy before age 45.
        - Premature menopause occurs if hysterectomy before age 40.
        """

        ppl = self.sim.people
    
        # Natural menopause based on age
        natural_meno = (ppl.female & ~self.menopausal & (ppl.age >= self.age_menopause))
        self.menopausal[natural_meno] = True
        self.age_at_menopause[natural_meno] = ppl.age[natural_meno]
    
        # Hysterectomy-based early/premature menopause
        new_hyst = (ppl.female & self.hyst & ~self.menopausal)
    
        # Determine who is <45 or <40 at hysterectomy
        early = new_hyst & (ppl.age < 45)
        premature = new_hyst & (ppl.age < 40)
    
        # Update all relevant states
        self.menopausal[early] = True
        self.early_meno[early] = True
        self.premature_meno[premature] = True
        self.age_at_menopause[early] = ppl.age[early]
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
            ss.Result('hiud_prev', scale=False, label="Prevalence of IUD usage"),
            ss.Result('early_meno_prev', scale=False, label="Early menopause prevalence"),
            ss.Result('premature_meno_prev', scale=False, label="Premature menopause prevalence"),
        ]
        self.define_results(*results)
        return

    def _get_uids(self, upper_age=None):
        """ Get uids of females younger than upper_age """
        people = self.sim.people
        if upper_age is None: upper_age = 1000
        within_age = people.age < upper_age
        return (within_age & people.female).uids

    def set_mens_states(self, upper_age=None):
        """ Set menstrual hygiene states """
        f_uids = self._get_uids(upper_age=upper_age)
        self.age_menses[f_uids] = self.pars.age_menses.rvs(f_uids)
        self.age_menopause[f_uids] = self.pars.age_menopause.rvs(f_uids)
        self.hmb_prone[f_uids] = self.pars.p_hmb_prone.rvs(f_uids)
        return

    @property
    def menstruating(self):
        return self.sim.people.female & ~self.menopausal & (self.sim.people.age >= self.age_menses)

    @property
    def hmb_sus(self):
        return self.menstruating & self.hmb_prone & ~self.hmb

    @property
    def lt40(self):
        return self.sim.people.age < 40

    def init_post(self):
        """ Initialize with sim properties """
        super().init_post()

        # Set initial menstrual states
        self.set_mens_states()
        self.set_hiud()
        self.set_hmb(self.hmb_sus.uids)

        return

    def _logistic(self, uids, pars):
        """ Calculate logistic regression probabilities """
        intercept = -np.log(1/pars.base-1)
        rhs = np.full_like(uids, fill_value=intercept, dtype=float)

        # Add all covariates
        for term, val in pars.items():
            if term != 'base':
                rhs += val * getattr(self, term)[uids]

        # Calculate the probability
        return 1 / (1+np.exp(-rhs))

    def set_hiud(self):
        """ Set who will use a hormonal IUD """
        self.hiud[:] = False  # Reset the state - TODO, should not reset every step!!!
        hiud_sus = self.menstruating.uids & ~self.hiud
        has_hiud = self.pars.p_hiud.filter(hiud_sus)
        self.hiud[has_hiud] = True
        return

    def assign_iud_types(self, ppl, p_hiud=0.5):
        """
        NOT FUNCTIONAL
        Among agents with method == IUD, assign hormonal vs copper IUDs.

        Args:
        ppl: the population object with .method and other arrays
        p_hiud: probability of having a hormonal IUD (default 0.5)
        """
        # Get index of the IUD method
        iud_method_idx = self.get_method_by_label("IUDs").idx

        # Find users assigned to IUD
        iud_users = np.where(ppl.method == iud_method_idx)[0]

        # Randomly assign to hormonal IUD vs copper IUD
        assigned_hiud = iud_users[np.random.rand(len(iud_users)) < p_hiud]
        assigned_cu_iud = np.setdiff1d(iud_users, assigned_hiud)

        # Set flags (make sure these arrays exist in ppl!)
        ppl.hiud[assigned_hiud] = True
        ppl.cu_iud[assigned_cu_iud] = True

        return assigned_hiud, assigned_cu_iud

    def set_hmb(self, uids):
        """ Set who will experience heavy menstrual bleeding (HMB) """
        # Calculate the probability of HMB
        p_hmb = self._logistic(uids, self.pars.hmb_pred)
        self._p_hmb.set(0)
        self._p_hmb.set(p_hmb)
        has_hmb = self._p_hmb.filter(uids)
        self.hmb[has_hmb] = True
        return

    def step(self):
        """ Updates for this timestep """
        self.set_mens_states(upper_age=self.t.dt)
        mens_uids = self.menstruating.uids
        self.hmb[:] = False  # Reset

        # Set IUD usage
        self.set_hiud()
        # self.assign_iud_types()

        # Update HMB
        self.set_hmb(self.hmb_sus.uids)

        # Set non-permanent sequalae of HMB
        for seq, p in self.pars.hmb_seq.items():
            old_attr = getattr(self, seq)
            old_attr[:] = False  # Reset the state
            setattr(self, seq, old_attr)  # Update the state
            attr_dist = getattr(self, f'_p_{seq}')
            attr_dist.set(0)

            # Calculate the probability of the sequelae
            p_val = self._logistic(mens_uids, p)
            attr_dist = getattr(self, f'_p_{seq}')
            attr_dist.set(p_val)
            has_attr = attr_dist.filter(mens_uids)
            new_attr = getattr(self, seq)
            new_attr[has_attr] = True
            setattr(self, seq, new_attr)

        # Set hysterectomy state
        hyst_sus = (self.menstruating & ~self.hyst).uids
        p_hyst = self._logistic(hyst_sus, self.pars.hyst)
        self._p_hyst.set(0)
        self._p_hyst.set(p_hyst)
        has_hyst = self._p_hyst.filter(hyst_sus)
        self.hyst[has_hyst] = True

        # # Disrupt education
        # TODO, remove this - it's done in the education module.
        # self.disrupt_education(self.sim.people, prob_disrupt=self.pars.hmb_seq.education.base)

        return

    def step_state(self):
        """ Updates for this timestep """
        return

    def update_results(self):
        super().update_results()
        ti = self.ti
        def count(arr): return np.count_nonzero(arr)
        def cond_prob(a, b): return sc.safedivide(count(a & b), count(b))
        for res in ['hmb', 'poor_mh', 'anemic', 'pain', 'hyst', 'hiud', 'early_meno', 'premature_meno']:
            self.results[f'{res}_prev'][ti] = cond_prob(getattr(self, res), self.menstruating)
        return

    def update(self, ppl):
        """
        NOT FUNCTIONAL
        TODO, see if we need this - suspect not since HMB status is assigned in set_hmb based on a prediction model
        """
        self.start_heavy_bleed(ppl)  # check for acquiring heavy bleed (may not have the data for this)
        self.stop_heavy_bleed(ppl)  # check for stopping heavy bleed (again, may not have the data for this)
        
        return
    

def disrupt_education(self, ppl, prob_disrupt=0.5): 
    """
    Probabilistically disrupt education due to heavy menstrual bleeding (HMB).
    Args:
        ppl: population object
        prob_disrupt: probability that HMB leads to school interruption (default 0.5)
    """

    # Get the disruption probability from parameters
    prob_disrupt = self.pars.hmb_seq.education.base

    # Find eligible students
    students = ppl.filter(
        ppl.female &
        ppl.edu_started & ~ppl.edu_completed & ~ppl.edu_dropout & ~ppl.edu_interrupted
    )

    # Subset: students currently menstruating and experiencing HMB
    hmb_students = students[ppl.hmb[students] & self.menstruating[students]]

    # Probabilistically assign school interruption
    will_disrupt = fpu.binomial_arr(prob_disrupt, len(hmb_students))
    disrupted_uids = hmb_students[will_disrupt]

    # Set interruption flag
    ppl.edu_interrupted[disrupted_uids] = True

    # Optional: count months of disruption
    if not hasattr(ppl, "edu_disruption_count"):
        ppl.edu_disruption_count = np.zeros(len(ppl), dtype=int)
    ppl.edu_disruption_count[disrupted_uids] += 1

    return

    
# ---------------- TEST ----------------

if __name__ == '__main__':

    mens = Menstruation()

    from education import Education
    objective_data = pd.read_csv(f"data/kenya_objective.csv")
    attainment_data = pd.read_csv(f"data/kenya_initialization.csv")
    edu = Education(objective_data=objective_data, attainment_data=attainment_data)

    sim = fp.Sim(location='kenya', connectors=[mens, edu], start=2020, stop=2030)
    sim.run(verbose=1/12)

    # Plot education
    import pylab as pl
    t = sim.results.education.timevec
    fig, axes = pl.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()

    res_to_plot = ['mean_attainment', 'mean_objective', 'prop_completed', 'prop_in_school', 'prop_dropped']
    sc.options(fontsize=16)

    for i, res in enumerate(res_to_plot):
        ax = axes[i]
        r0 = sim.results.education[res]
        ax.plot(t, r0)
        ax.set_title(res)

    all_props = [sim.results.education.prop_in_school,
                 sim.results.education.prop_completed,
                 sim.results.education.prop_dropped]

    ax = axes[-1]
    ax.stackplot(t, all_props, labels=['In school', 'Completed', 'Dropped'], alpha=0.8)
    ax.set_title('All AGYW')
    ax.legend()

    sc.figlayout()
    pl.show()





    
    
    
    
    
    
    
    
    
    
        
        
        
        

