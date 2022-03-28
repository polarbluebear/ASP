# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
import scipy.integrate as spint
import pyfeng as pf
from . import normal
from . import bsm

'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0, time_steps=1_000, n_samples=10_000):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.time_steps = time_steps
        self.n_samples = n_samples
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        Use self.bsm_model.impvol() method
        '''
        price = self.price(strike, spot, texp, sigma)
        vol = self.bsm_model.impvol(price, strike, spot, texp)
        return vol
  
    
    def price(self, strike, spot, texp=None, sigma=None, cp=1, time_steps=1_000, n_samples=10_000):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        np.random.seed(12345)
        div_fac = np.exp(-texp * self.divr)
        disc_fac = np.exp(-texp * self.intr)
        forward = spot / disc_fac * div_fac
        
        if sigma is None:
            sigma = self.sigma
    
        self.time_steps = time_steps         # number of time steps of MC
        self.n_samples = n_samples          # number of samples of MC

        # Generate correlated normal random variables W1, Z1
        z = np.random.normal(size=(self.n_samples, self.time_steps))
        x = np.random.normal(size=(self.n_samples, self.time_steps))
        w = self.rho * z + np.sqrt(1-self.rho**2) * x

        path_size = np.zeros([self.n_samples, self.time_steps + 1])   
        delta_tk = texp / self.time_steps                      
        log_sk = np.log(spot) * np.ones_like(path_size)      # log price
        sk = spot * np.ones_like(path_size)              # price
        sigma_tk = self.sigma * np.ones_like(path_size)       # sigma
        for i in range(self.time_steps):
            log_sk[:, i+1] = log_sk[:, i] + sigma_tk[:, i] * np.sqrt(delta_tk) * w[:, i] - 0.5 * (sigma_tk[:, i]**2) * delta_tk
            sigma_tk[:, i+1] = sigma_tk[:, i] * np.exp(self.vov * np.sqrt(delta_tk) * z[:, i] - 0.5 * (self.vov**2) * delta_tk)
            sk[:, i+1] = np.exp(log_sk[:, i+1])

        price = np.zeros_like(strike)
        
        for j in range(len(strike)):
            price[j] = np.mean(np.maximum(sk[:, -1] - strike[j], 0))
        return  disc_fac * price


'''
MC model class for Beta=0
'''
class ModelNormalMC: 
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0, time_steps=1_000, n_samples=10_000):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.time_steps = time_steps
        self.n_samples = n_samples
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol.
        Use self.normal_model.impvol() method        
        '''
        price = self.price(strike, spot, texp, sigma)
        vol = self.normal_model.impvol(price, strike, spot, texp)
        return vol
        
    def price(self, strike, spot, texp=None, sigma=None, cp=1, time_steps=1_000, n_samples=10_000):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        np.random.seed(12345)
        div_fac = np.exp(-texp * self.divr)
        disc_fac = np.exp(-texp * self.intr)
        forward = spot / disc_fac * div_fac
        
        if sigma is None:
            sigma = self.sigma
    
        self.time_steps = time_steps         # number of time steps of MC
        self.n_samples = n_samples          # number of samples of MC

        # Generate correlated normal random variables W1, Z1
        z = np.random.normal(size=(self.n_samples, self.time_steps))
        x = np.random.normal(size=(self.n_samples, self.time_steps))
        w = self.rho * z + np.sqrt(1-self.rho**2) * x

        path_size = np.zeros([self.n_samples, self.time_steps + 1])   
        delta_tk = texp / self.time_steps                      
        
        sk = spot * np.ones_like(path_size)              # price
        sigma_tk = self.sigma * np.ones_like(path_size)       # sigma
        for i in range(self.time_steps):
            sk[:, i+1] = sk[:, i] + sigma_tk[:, i] * np.sqrt(delta_tk) * w[:, i]
            sigma_tk[:, i+1] = sigma_tk[:, i] * np.exp(self.vov * np.sqrt(delta_tk) * z[:, i] - 0.5 * (self.vov**2) * delta_tk)
              
        price = np.zeros_like(strike)
        
        for j in range(len(strike)):
            price[j] = np.mean(np.maximum(sk[:, -1] - strike[j], 0))
        return  disc_fac * price

'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=1.0, intr=0, divr=0, time_steps=1_000, n_samples=10_000):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.time_steps = time_steps
        self.n_samples = n_samples
        self.bsm_model = pf.Bsm(sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None):
        ''''
        should be same as bsm_vol method in ModelBsmMC (just copy & paste)
        '''
        price = self.price(strike, spot, texp, sigma)
        vol = self.bsm_model.impvol(price, strike, spot, texp)
        return vol
    
    def price(self, strike, spot, texp=None, cp=1, time_steps=1_000, n_samples=10_000):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        np.random.seed(12345)
        div_fac = np.exp(-texp * self.divr)
        disc_fac = np.exp(-texp * self.intr)
        forward = spot / disc_fac * div_fac

    
        self.time_steps = time_steps         # number of time steps of MC
        self.n_samples = n_samples          # number of samples of MC

        # Generate correlated normal random variables Z
        z = np.random.normal(size=(self.n_samples, self.time_steps))  

        delta_tk = texp / self.time_steps                       
        sigma_tk = self.sigma * np.ones([self.n_samples, self.time_steps+1])   
        
        for i in range(self.time_steps):
            sigma_tk[:, i+1] = sigma_tk[:, i] * np.exp(self.vov * np.sqrt(delta_tk) * z[:, i] - 0.5 * (self.vov ** 2) * delta_tk)

        I = spint.simps(sigma_tk * sigma_tk, dx=texp/self.time_steps) / (self.sigma**2)  # integrate by using Simpson's rule
        
        spot_cond = spot * np.exp(self.rho * (sigma_tk[:, -1] - self.sigma) / self.vov - (self.rho*self.sigma)**2 * texp * I / 2)
        vol = self.sigma * np.sqrt((1 - self.rho**2) * I)

        price = np.zeros_like(strike)
        
        for j in range(len(strike)):
            price[j] = np.mean(bsm.price(strike[j], spot_cond, texp ,vol))
            
        return  disc_fac * price

'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    vov, rho = 0.0, 0.0
    sigma, intr, divr = None, None, None
    normal_model = None
    
    def __init__(self, sigma, vov=0, rho=0.0, beta=0.0, intr=0, divr=0, time_steps=1_000, n_samples=10_000):
        self.sigma = sigma
        self.vov = vov
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.time_steps = time_steps
        self.n_samples = n_samples
        self.normal_model = pf.Norm(sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None):
        ''''
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        price = self.price(strike, spot, texp, sigma)
        vol = self.normal_model.impvol(price, strike, spot, texp)
        return vol
        
        
    def price(self, strike, spot, texp=None, cp=1, time_steps=1_000, n_samples=10_000):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        np.random.seed(12345)
        div_fac = np.exp(-texp * self.divr)
        disc_fac = np.exp(-texp * self.intr)
        forward = spot / disc_fac * div_fac

        self.time_steps = time_steps         # number of time steps of MC
        self.n_samples = n_samples          # number of samples of MC

        # Generate correlated normal random variables Z
        z = np.random.normal(size=(self.n_samples, self.time_steps))  

        delta_tk = texp / self.time_steps                       
        sigma_tk = self.sigma * np.ones([self.n_samples, self.time_steps+1])   
        
        for i in range(self.time_steps):
            sigma_tk[:, i+1] = sigma_tk[:, i] * np.exp(self.vov * np.sqrt(delta_tk) * z[:, i] - 0.5 * (self.vov ** 2) * delta_tk)

        I = spint.simps(sigma_tk * sigma_tk, dx=texp/self.time_steps) / (self.sigma**2)  # integrate by using Simpson's rule
        
        spot_cond = spot + self.rho * (sigma_tk[:, -1] - self.sigma) / self.vov
        vol = self.sigma * np.sqrt((1 - self.rho**2) * I)

        price = np.zeros_like(strike)
        
        for j in range(len(strike)):
            price[j] = np.mean(normal.price(strike[j], spot_cond, texp ,vol))
          
        return  disc_fac * price
