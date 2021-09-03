def recall_powerlaw_fits_to_full_models():
    """fits may be recomputed by evaluating the .ipynb associated with fitting powerlaws to the full models.
    here, w=M*q**m, and Delta_X is the maximum disagreement one could expect to observe with 95% confidence.
    here, we observe Delta_X concerns disagreements between statistically independent measurements of X.

    Example Usage:
    wjr=recall_powerlaw_fits_to_full_models()
    print(*wjr)
    """
    # Recall powerlaw fits to full models
    # Fenton-Karma(PBC)
    m, Delta_m, M, Delta_M = 1.8772341309722325, 0.02498750277237229, 5.572315674840435, 0.3053120355191732
    wjr={
        'fk_pbc':{'m':m, 'Delta_m':Delta_m, 'M':M, 'Delta_M':Delta_M}
    }
    # Luo-Rudy(PBC)
    m, Delta_m, M, Delta_M = 1.6375562704001745, 0.017190912126700632, 16.73559858353835, 0.8465090320196467
    wjr['lr_pbc']={'m':m, 'Delta_m':Delta_m, 'M':M, 'Delta_M':Delta_M}
    return wjr
