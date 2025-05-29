"""
Heavy Menstrual Bleeding
Adds an agent state to proxy heavy menstrual bleeding and initializes state 
"""
import fpsim as fp
import numpy as np
import pandas as pd
import sciris as sc

import fpsim.defaults as fpd
import fpsim.locations as fplocs

class Menstruation:
    
    '''Create a class to handle menstruation related events'''
    
    def initialize(self, ppl):
        """ Initialize with people """
        menstruation_dict = self.pars
        
        # Initialize with an age at menarche
        menarche_age = random.random(11,15)
        
        #Note: I don't want to hard-code this, so need to decide if there's enough data out there to create a range and/or distribution..
        
        # Set location-specific probability of HMB
        prob_heavy_bleed = ##data placeholder
        
        # Set initial anemia levels, location specific
        prob_anemia = ##data placeholder
        
        return

    def update(self, ppl):
        self.start_heavy_bleed(ppl) #check for acquiring heavy bleed (may not have the data for this)
        self.stop_heavy_bleed(ppl) #check for stopping heavy bleed (again, may not have the data for this)
        
        return
    
    def amenorrhea_pref(self,ppl): 
        """ 
        Time-invariant amenorrhea preferences which will impact contraceptive method choice.
        """
        
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
        probs_disrupt = #placeholder
        HMB_students.ed_disrupted = #binomial trial placeholder 
        #psuedo code placeholder: If HMB_students.ed_disrupted == 1, then HMB_students.interrupt_education = TRUE
        
        
        ##TO DO: set # of months disrupted to impact attainment/dropout
        ##TO DO: make temporary per timestep (there must be a quick way to do this and might need to separate from existing interruption mechanism)
        return
    
    def iron_deficient(self,ppl):
        """
        Determine whether women with HMB become iron deficient. Will impact risk of anemia.
        """
        ## data placeholder
        
        return
    
    def anemia(self, ppl): 
        
        """
        Increased risk of anemia due to HMB. 
        """
        
        ## data placeholder
        
        return
    
    def hormonal_method_prefs(self, ppl):
        
        """
        Link HMB state to increased probability of choosing a hormonal method to help reduce heavy bleeding
        Hormonal methods that reduce bleeding: Hormonal IUD; Oral contraceptive pill; Implant; Injectable
        """
        
        return
    
    def hysterectomy(self, ppl):
        """ 
        Establish the population-wide prevalence of hysterectomy. 
        Also links tubal litigation to increased odds of hysterectomy. 
        For India, draws on data from Prusty et al (2018): https://reproductive-health-journal.biomedcentral.com/articles/10.1186/s12978-017-0445-8
        """
        
        ## data placeholder
        
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
    
    
    
    
    
    
    
    
    
    
        
        
        
        

