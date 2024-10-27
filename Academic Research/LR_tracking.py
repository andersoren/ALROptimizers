#%%
import pandas as pd
import matplotlib.pyplot as plt

# Set the DPI and figure size
dpi = 100
width_in_inches = 400 / dpi
height_in_inches = 300 / dpi

# Create the figure
fig = plt.figure(figsize=(width_in_inches, height_in_inches), dpi=dpi)

#%%
OneCycle = pd.read_csv(r"OneCycleLR-RawData.csv")

plt.plot(OneCycle["scheduler_steps"], OneCycle["optimizer: Adam - Global learning rate"])
plt.xlabel("Scheduler steps during training")
plt.ylabel("Global learning-rate")
plt.title("Global learning rate for OneCyCleLR scheme")
plt.show()
# %%

AdamUpd_mean = pd.read_csv(r"AdamUpd-Mean-Bad.csv")
AdamUpd_std = pd.read_csv(r"AdamUpd-Std-Bad.csv")
plt.errorbar(AdamUpd_mean["Step"], AdamUpd_mean["optimizer: AdamUpd - Indiv. learning rate mean"], \
             fmt='o', yerr=AdamUpd_std["optimizer: AdamUpd - Indiv. learning rate std"], capsize=5)
plt.xlabel("Scheduler steps during training")
plt.ylabel("Individual learning-rate average")
plt.title(r"Individual learning-rates for Adam-Upd ($\eta_-=0.5$)")
plt.show()


# %%
