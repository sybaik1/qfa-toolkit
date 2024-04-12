# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import json
from matplotlib import pyplot as plt


# %%
def load_data(p: int, k: int) -> dict[dict[str, float], dict[str, float]]:
    with open(f"../../results/{p}_{k}_observed.json", "r") as f:
        line = f.readline()
        idx = line.index(']')
        observed_acceptance, observed_rejection = tuple(json.loads(line[:idx+1]))
        target_rejection, target_acceptance = tuple(json.loads(line[idx+1:]))
        observed = {
            'acceptance': observed_acceptance,
            'undefined': 1 - observed_acceptance - observed_rejection,
            'rejection': observed_rejection,
        }
        target = {
            'acceptance': target_acceptance,
            'rejection': target_rejection,
        }
        return {'observed': observed, 'tareget': target}


# %%
ps = [3, 5]
data_dict = {
    p: {k: load_data(p, k) for k in range(1, 11)}
    for p in ps
}

# %%
p = 5
labels = data_dict[p].keys()
observed_acceptances = np.array([data_dict[p][label]['observed']['acceptance'] for label in labels])
observed_rejections = np.array([data_dict[p][label]['observed']['rejection'] for label in labels])
xs = observed_acceptances


# %%
print(xs)
plt.bar(labels, xs)

# %%
