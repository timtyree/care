#! /usr/bin/env python
# fit_poisson.py
# Tim Tyree
# 8.12.2020
# the maximum-likelihood estimator for the parameter of the poissonian distribution is the arithmetic mean.
import numpy as np
from scipy.stats import poisson
def _get_mean(array):
	if type(array) is not type(np.array([0])):
		raise Exception("array must be a numpy.array!")
	return np.mean(array)

def _get_poisson_pdf(mean):
	return poisson(mu=float(mean))

def fit_poisson(array):
	mean = _get_mean(array)
	return _get_poisson_pdf(mean)