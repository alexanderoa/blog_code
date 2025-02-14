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

def many_cuped_sims(
    n_sims = 100,
    sample_size = 500,
    t_proportion = 0.5,
    seed = 123
):
    '''
    Simulates draws from cuped_generator and produces three different estimates

    :param n_sims: number of datasets to simulate
    :param sample_size: number of samples in each dataset
    :param seed: seed for meta_rng

    :return simple: estimates for simple regression (i.e. diff-in-means)
    :return cuped: estimates for CUPED
    :return adjust: estimates for regression-adjustment
    '''
    meta_rng = np.random.default_rng(seed=seed)
    simple = np.zeros(n_sims)
    cuped = np.zeros(n_sims)
    adjust = np.zeros(n_sims)
    for i in range(n_sims):
        data = cuped_generator(
            seed=meta_rng.integers(1,1e10),
            sample_size = sample_size,
            t_proportion = t_proportion)

        cuped_lm = sm.OLS(data['Post_trigger'], data['Pre_normalized']).fit()
        theta = cuped_lm.params[0]
        data['Post_cuped'] = data['Post_trigger'] - theta*data['Pre_normalized']

        reg = smf.ols("Post_trigger ~ Treatment",data).fit()
        simple[i] = reg.params['Treatment']

        cuped_reg = smf.ols("Post_cuped ~ Treatment", data).fit()
        cuped[i] = cuped_reg.params['Treatment']

        reg_adj = smf.ols(
            formula="Post_trigger ~ Treatment + Pre_trigger + Treatment:Pre_normalized",
            data=data).fit()
        adjust[i] = reg_adj.params['Treatment']
    return simple, cuped, adjust