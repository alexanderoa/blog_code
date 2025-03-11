from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

def nonlinear_generator(
    sample_size = 500,
    effect = 0.5,
    t_proportion = 0.5,
    x_coef = 0.75,
    seed = 123,
):
    rng = np.random.default_rng(seed=seed)
    x = rng.uniform(low=-10, high=10, size = (sample_size,))
    t = rng.binomial(n=1, p=t_proportion, size=(sample_size,))
    xt = x_coef*(x**2/20-0.2)*(x/5)
    y = effect*t + xt + rng.normal(scale = 1, size=(sample_size,))
    x_centered = x - np.mean(x)

    data = pd.DataFrame(
        {
            "Y" : y,
            "X" : x,
            "X_centered" : x_centered,
            "T" : t
        }
    )
    return data

def mlrate_df(
    data,
    model,
    n_splits = 2
):
    kf = KFold(
        n_splits = n_splits, 
        shuffle=True)
    kf.get_n_splits(data)
    final = data.copy()
    combined = np.zeros(data.shape[0])
    for i, (train_idx, test_idx) in enumerate(kf.split(data)):
        train = data.loc[train_idx]
        test = data.loc[test_idx]
        g = model.fit(train[['X']].to_numpy(), train['Y'].to_numpy())
        gpreds = g.predict(test[['X']].to_numpy())
        combined[test_idx] = gpreds

    final["G"] = combined
    final["G_centered"] = combined - np.mean(combined)

    return final

def nonlinear_variance_sim(
    n_sims = 100,
    seed = 123
):
    meta_rng = np.random.default_rng(seed=seed)
    diff = np.zeros(n_sims)
    adjust = np.zeros(n_sims)
    mlrate = np.zeros(n_sims)

    for i in range(n_sims):
        sim_seed = meta_rng.integers(low=0, high = 100000)
        data = nonlinear_generator(seed=sim_seed)

        diff_reg = smf.ols("Y~T", data=data).fit(cov_type='HC2')
        diff[i] = diff_reg.params['T']

        adj_reg = smf.ols(
            "Y ~ X + T + X_centered:T",
            data = data
        ).fit(cov_type='HC2')
        adjust[i] = adj_reg.params['T']

        final = mlrate_df(
            data = data,
            model = GradientBoostingRegressor()
        )
        ml_reg = smf.ols(
            "Y ~ T + G + T:G_centered",
            data = final
        ).fit(cov_type="HC2")
        mlrate[i] = ml_reg.params['T']
    diff_df = pd.DataFrame({'Estimate':diff, "Method": ['Diff-in-Means']*n_sims})
    adj_df = pd.DataFrame({'Estimate':adjust, "Method": ['Adjustment']*n_sims}) 
    ml_df = pd.DataFrame({'Estimate':mlrate, "Method": ['MLRATE']*n_sims})
    df = pd.DataFrame()
    df = df.append([diff_df, adj_df, ml_df])
    return df