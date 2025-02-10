import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def cuped_generator(
    sample_size = 500,
    t_proportion = 0.5,
    effect_size = 0.5,
    seed = 123
):
    """
    Generates data for CUPED blog post
     - Simulated as an outcome with pre-exposure and post-exposure measurements
     - Pre-exposure simulated as Normal(mean-5,sd=2)
     - Post-exposure simulated as (Pre + treatment*effect_size + noise)

    :param sample_size: sample size for simulated dataset
    :param t_proportion: proportion of samples assigned to treatment
    :param effect_size: size of the effect of treatment (additive)
    :param seed: seed for numpy random number generator

    :return df: data frame with simulated data
    """
    rng = np.random.default_rng(seed=seed)

    t = rng.binomial(n=1,p=t_proportion,size=(sample_size,))
    pre = rng.normal(loc=5, scale=2, size=(sample_size,))
    post = (
        pre + 
        rng.normal(loc=3, size=(sample_size,)) + 
        t*effect_size
    )
    pre_normal = pre - np.mean(pre)
    df = pd.DataFrame(
        {
            "Treatment": t,
            "Pre_trigger" : pre,
            "Post_trigger" : post,
            "Pre_normalized" : pre_normal
        }
    )
    return df