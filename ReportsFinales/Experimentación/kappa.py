#!/usr/bin/env python2

import numpy as np

def fleiss_kappa(table):
    '''Fleiss' kappa multi-rater agreement measure

    Parameters
    ----------
    table : array_like, 2-D
        assumes subjects in rows, and categories in columns

    Returns
    -------
    kappa : float
        Fleiss's kappa statistic for inter rater agreement

    Notes
    -----
    coded from Wikipedia page
    http://en.wikipedia.org/wiki/Fleiss%27_kappa

    no variance or tests yet

    '''
    table = 1.0 * np.asarray(table)   #avoid integer division
    n_sub, n_cat =  table.shape
    n_total = table.sum()
    n_rater = table.sum(1)
    n_rat = n_rater.max()
    #assume fully ranked
    assert n_total == n_sub * n_rat

    #marginal frequency  of categories
    p_cat = table.sum(0) / n_total

    table2 = table * table
    p_rat = (table2.sum(1) - n_rat) / (n_rat * (n_rat - 1.))
    p_mean = p_rat.mean()

    p_mean_exp = (p_cat*p_cat).sum()

    if(p_mean_exp ==1):
    	kappa = 1

    else: 
    	kappa = (p_mean - p_mean_exp) / (1- p_mean_exp)

    	
    return kappa


