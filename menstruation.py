"""
Heavy Menstrual Bleeding
Adds an agent state to proxy heavy menstrual bleeding and initializes state 
"""
import numpy as np
import sciris as sc
import starsim as ss



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

            # --- Prob of hormonal IUD
            # The probability of IUD usage is set within FPsim, so this parameter just
            # determines whether each woman is a hormonal or non-hormonal IUD user
            p_hiud=ss.bernoulli(p=0.17),

            # --- HMB prediction
            # Proportion of menstruating women who experience HMB (sans interventions)
            p_hmb_prone=ss.bernoulli(p=0.486),  
            
            
            hmb_pred=sc.objdict(  # Parameters for HMB prediction
                # Baseline odds that those prone to HMB will experience it this timestep
                # This is converted to an intercept in the logistic regression: -np.log(1/base-1)
                base=0.95,
                pill = -np.log(1/((1 - 0.25*0.312) * 0.95) -1) - np.log(1/0.95 -1),
                hiud = -np.log(1/((1-0.312) * 0.95) -1) - np.log(1/0.95 -1),
                txa = -np.log(1/((1 - 0.5*0.312) * 0.95) -1) - np.log(1/0.95 -1),
            ),

            # Non-permanent sequelae of HMB
            hmb_seq=sc.objdict(
                poor_mh=sc.objdict(  # Parameters for poor menstrual hygiene
                    base = 0.4,  # Intercept for poor menstrual hygiene
                    hiud = -0.5,  # Effect of hormonal IUD on poor menstrual hygiene - placeholder 
                    pill = -0.5, # Effect of pill on poor menstrual hygiene - placeholder
                    txa = -0.5, # Effect of TXA on poor menstrual hygiene - placeholder
                ),
                #anemic=sc.objdict(  # Parameters for anemia
                #    base = 0.01,  # Baseline probability of anemia
                #    hmb = 1.5,  # Effect of HMB on anemia - placeholder : prob of anemia is 1/(1+np.exp(-(-np.log(1/0.01 -1)+1.5))) = 0.0433
                #),
                anemic=sc.objdict(  # Parameters for anemia
                    # This is converted to an intercept in the logistic regression: -np.log(1/base-1)
                    base = 0.18,  # Baseline probability of anemia
                    hmb = -np.log(1/0.35 - 1) + np.log(1/0.18 - 1),
                ),
                pain=sc.objdict(  # Parameters for menstrual pain
                    base = 0.1,  # Baseline probability of menstrual pain
                    hmb = 1.5,  # Effect of HMB on menstrual pain - placeholder: prob of pain is 1/(1+np.exp(-(-np.log(1/0.1 -1)+1.5))) = 0.332
                    hiud = -0.5,  # Effect of hormonal IUD on menstrual pain - placeholder
                    pill = -0.5, # Effect of pill on menstrual pain - placeholder
                    txa = -0.5, # Effect of TXA on menstrual pain - placeholder
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

            # Contraceptive methods and other HMB prevention methods
            ss.BoolState('pill', label="Using hormonal pill"),
            ss.BoolState('hiud', label="Using hormonal IUD"),
            ss.BoolState('txa', label="Using tranexamic acid"),
            ss.BoolState('hiud_prone', label="Prone to use hormonal IUD, if using IUD"),
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
            ss.Result('hiud_prev', scale=False, label="Prevalence of hIUD usage"),
            ss.Result('pill_prev', scale=False, label="Prevalence of pill usage"),
            ss.Result('early_meno_prev', scale=False, label="Early menopause prevalence"),
            ss.Result('premature_meno_prev', scale=False, label="Premature menopause prevalence"),
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
        self.hiud_prone[f_uids] = self.pars.p_hiud.rvs(f_uids)
        
        return


    def init_post(self):
        """ Initialize with sim properties """
        super().init_post()

        # Set initial menstrual states
        self.set_mens_states()
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


    def set_hmb(self, uids):
        """ Set who will experience heavy menstrual bleeding (HMB) """
        # Calculate the probability of HMB (based on interventions)
        p_hmb = self._logistic(uids, self.pars.hmb_pred)
        self._p_hmb.set(0)
        self._p_hmb.set(p_hmb)
        # filter returns True if p_hmb > 0.5
        has_hmb = self._p_hmb.filter(uids)
        # set hmb among those uids filtered above
        self.hmb[has_hmb] = True
        return


    def step(self):
        """ Updates for this timestep """
                
        # Set menstruating states
        self.set_mens_states(upper_age=self.t.dt)  # for new agents
        self.step_states()  # for existing agents

        mens_uids = self.menstruating.uids
        self.hmb[:] = False  # Reset

        # Update HMB
        self.set_hmb(self.hmb_sus.uids)

        # Set non-permanent sequalae of HMB
        for seq, p in self.pars.hmb_seq.items():
            old_attr = getattr(self, seq)
            old_attr[:] = False  # Reset the state
            # Update the state
            #setattr(self, seq, old_attr)  
            self.setattribute(seq, old_attr)
            attr_dist = getattr(self, f'_p_{seq}')
            attr_dist.set(0)

            # Calculate the probability of the sequelae
            p_val = self._logistic(mens_uids, p)
            attr_dist = getattr(self, f'_p_{seq}')
            attr_dist.set(p_val)
            has_attr = attr_dist.filter(mens_uids)
            new_attr = getattr(self, seq)
            new_attr[has_attr] = True
            # Update the state
            #setattr(self, seq, new_attr)
            self.setattribute(seq, new_attr)

        # Set hysterectomy state
        hyst_sus = (self.menstruating & ~self.hyst).uids
        p_hyst = self._logistic(hyst_sus, self.pars.hyst)
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
        self.hmb_sus[:] = self.menstruating & self.hmb_prone & ~self.hmb

        # Contraceptive methods
        pill_idx = cm.get_method_by_label('Pill').idx
        iud_idx = cm.get_method_by_label('IUDs').idx
        self.pill[:] = ppl.fp.method == pill_idx
        self.hiud[:] = (ppl.fp.method == iud_idx) & self.hiud_prone

        return


    def update_results(self):
        super().update_results()
        ti = self.ti
        def count(arr): return np.count_nonzero(arr)
        def cond_prob(a, b): return sc.safedivide(count(a & b), count(b))
        for res in ['hmb', 'poor_mh', 'anemic', 'pain', 'hiud', 'hyst', 'early_meno', 'premature_meno', 'pill']:
            self.results[f'{res}_prev'][ti] = cond_prob(getattr(self, res), self.menstruating)
        for res in ['hyst', 'early_meno', 'premature_meno']:
            self.results[f'{res}_prev'][ti] = cond_prob(getattr(self, res), self.post_menarche)
        return



