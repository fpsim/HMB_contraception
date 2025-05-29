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
        
        return

    def update(self, ppl):
        self.start_heavy_bleed(ppl) #check for acquiring heavy bleed (may not have the data for this)
        self.stop_heavy_bleed(ppl) #check for stopping heavy bleed (again, may not have the data for this)
        
        return
    
    def disrupt_education(self, ppl): 
        """
        Temporarily disrupt education due to heavy menstrual bleeding. 
        This disruption can interrupt education progress if it occurs for multiple months in a row.
        """
        
        # Hinder education progression if a woman is pregnant and towards the end of the first trimester
        HMB_students = ppl.filter(ppl.heavy_bleed & (ppl.gestation == ppl.pars['end_first_tri']))
        # Disrupt education
        HMB_students.ed_disrupted = #placeholder for probability, may need to be assumption
        
        return
    
    
        
        
        
        

