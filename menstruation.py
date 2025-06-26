"""
Heavy Menstrual Bleeding
Adds an agent state to proxy heavy menstrual bleeding and initializes state 
"""
import fpsim as fp
import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss

import fpsim.defaults as fpd
import fpsim.locations as fplocs

class Menstruation(fp.Module):
    
    '''Create a class to handle menstruation related events'''
    
    def __init__(self, pars=None, name='menstruation', **kwargs):
        super().__init__(name=name)

        # Define parameters
        self.define_pars(
            unit='month',
            
        # Menses
        age_menses=ss.lognorm_ex(14, 3),  # Age of menarche
        age_menopause=ss.normal(50, 3),  # Age of menopause ##TODO: Allow for early menopause, generate a flag

        # HMB prediction
        p_hmb_prone=ss.bernoulli(p=0.4),  # Proportion of menstruating women who experience HMB (sans interventions)
        hmb_pred=sc.objdict(  # Parameters for HMB prediction
                            base=0.95,  # For those prone to HMB, probability they'll experience it this timestep
                            iud=-10,  # Effect of IUD on HMB - placeholder
                        ),

        # Non-permanent sequelae of HMB
        hmb_seq=sc.objdict(
        poor_mh=sc.objdict(  # Parameters for poor menstrual hygiene
                            base=0.4,  # Intercept for poor menstrual hygiene
                            iud=-0.5,  # Effect of IUD on poor menstrual hygiene - placeholder ##TODO: Allow for an effect of other hormonal contraception
                        ),
                        anemic=sc.objdict(  # Parameters for anemia
                            base=0.01,  # Baseline probability of anemia
                            hmb=1.5,  # Effect of HMB on anemia - placeholder
                        ),
                        pain=sc.objdict(  # Parameters for menstrual pain
                            base=0.1,  # Baseline probability of menstrual pain
                            hmb=1.5,  # Effect of HMB on menstrual pain - placeholder
                            iud=-0.5,  # Effect of IUD on menstrual pain - placeholder ##TODO: Other contraceptive methods 
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
                    ss.State('hmb_prone'),  # Prone to HMB
                    ss.State('hmb'),
                    ss.State('anemic'),
                    ss.State('poor_mh', label="Poor menstrual hygiene"),
                    ss.State('pain', label="Menstrual pain"),
                    ss.State('hyst', label="Hysterectomy"),
                    ss.State('hiud', label="Hormonal IUD usage"),
                    ss.FloatArr('age_menses', label="Age of menarche"),
                    ss.FloatArr('age_menopause', label="Age of menopause"),
                )

        return

    def early_menopause(self,ppl):
        
        """
        Define age cut off to be considered in early menopause. 
        Hysterectomy before that age will put women into early menopause.
        """
        
        early_age = 45
        premature_age = 40
        
        ##pseudo code: if age < early_age & hysterectomy == TRUE, then early_menopause
        ##pseudo code: if age < premature_age & hysterectomy == TRUE, then premature_menopause
        
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
        ppl = self.sim.people
        return ppl.female & (ppl.age > self.age_menses) & (ppl.age < self.age_menopause)

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
        self.set_iud()
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

    def assign_iud_types(self, ppl, p_hiud=0.5):
        """
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

                # Set hormonal IUD usage
                self.assign_iud_types()

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

                return

    def step_state(self):
                """ Updates for this timestep """
                return

    def update_results(self):
                super().update_results()
                ti = self.ti
                def count(arr): return np.count_nonzero(arr)
                def cond_prob(a, b): return sc.safedivide(count(a & b), count(b))
                for res in ['hmb', 'poor_mh', 'anemic', 'pain', 'hyst', 'iud']:
                    self.results[f'{res}_prev'][ti] = cond_prob(getattr(self, res), self.menstruating)
                return



    def update(self, ppl):
        self.start_heavy_bleed(ppl) #check for acquiring heavy bleed (may not have the data for this)
        self.stop_heavy_bleed(ppl) #check for stopping heavy bleed (again, may not have the data for this)
        
        return
    
    def amenorrhea_pref(self,ppl): 
        """ 
        Time-invariant amenorrhea preferences which will impact contraceptive method choice.
        """
        ## TODO: Define distribution; link to increase in prob of pills vs. IUDs
        ## data placeholder
        
        return
    
    def disrupt_education(self, ppl): 
        """
        Temporarily disrupt education due to heavy menstrual bleeding. 
        This disruption can interrupt education progress if it occurs for multiple months in a row.
        """
        
        # Filter people who have not: completed education, dropped out or had their education interrupted
        students = ppl.filter((ppl.edu_started & ~ppl.edu_completed & ~ppl.edu_dropout & ~ppl.edu_interrupted))
        
        # Hinder education progression if a woman is pregnant and towards the end of the first trimester
        HMB_students = ppl.filter(ppl.students & ppl.heavy_bleed)
        # Disrupt education
        probs_disrupt = 0.5 #placeholder
        HMB_students.ed_disrupted = np.random.binomial(1, probs_disrupt, size=len(HMB_students))
       
        
        ##TO DO: set # of months disrupted to impact attainment/dropout
        ##TO DO: make temporary per timestep (there must be a quick way to do this and might need to separate from existing interruption mechanism)
        return
    
# ---------------- TEST ----------------
def test_menstruation_module():
    sim = ss.Sim(pop_size=1000, n_days=30)
    mod = Menstruation()
    sim.add(mod)
    sim.run()

    # Check that results were collected
    for res_key in ['hmb_prev', 'poor_mh_prev', 'anemic_prev', 'pain_prev', 'hyst_prev', 'iud_prev']:
        assert res_key in mod.results
        assert len(mod.results[res_key]) == sim.npts
        assert np.all(mod.results[res_key] >= 0)

    print("Menstruation module test passed.")

    



    
    
    
    
    
    
    
    
    
    
        
        
        
        

