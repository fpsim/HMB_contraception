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
        
        """ Initialize with an age at menarche"""
        menarche_age = random.random(11,15)
        

