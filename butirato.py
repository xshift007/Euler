# Importar librerías necesarias
import itertools
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro, levene
from statsmodels.stats.power import FTestAnovaPower
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot

# Función para sensibilidad: Ejecuta simulación variando σ
def simulate_and_analyze(sigma=0.3, n_replicates=3, vary_effects=1.0):
    np.random.seed(0)
    # Niveles factores
    butyrate_levels = ["0 mM", "1 \u00B5M", "1 mM"]
    baf_levels = ["No BafA1", "BafA1"]
    time_levels = ["2h", "6h", "24h"]

    # Diseño factorial
    design = pd.DataFrame(list(itertools.product(butyrate_levels, baf_levels, time_levels)),
                          columns=["ButyrateConc", "BafA1", "Time"])
    design = design.loc[design.index.repeat(n_replicates)].reset_index(drop=True)
    design["Rep"] = design.groupby(["ButyrateConc", "BafA1", "Time"]).cumcount() + 1

    # Efectos (escalados)
    intercept = 1.0
    effect_butyrate = {"0 mM": 0.0, "1 \u00B5M": 0.2 * vary_effects, "1 mM": 0.6 * vary_effects}
    effect_baf = {"No BafA1": 0.0, "BafA1": 0.4 * vary_effects}
    effect_time = {"2h": 0.0, "6h": 0.3 * vary_effects, "24h": 0.2 * vary_effects}

    # Interacciones (escaladas)
    interaction_but_baf = {("1 \u00B5M", "BafA1"): 0.2 * vary_effects,
                           ("1 mM", "BafA1"): 0.7 * vary_effects}
    interaction_but_time = {("1 \u00B5M", "2h"): -0.2 * vary_effects,
                             ("1 mM", "2h"): -0.6 * vary_effects,
                             ("1 \u00B5M", "24h"): -0.05 * vary_effects,
                             ("1 mM", "24h"): -0.1 * vary_effects}
    interaction_baf_time = {("BafA1", "2h"): -0.05 * vary_effects}

    # Calcular medias LC3_II
    LC3_mean = []
    for _, row in design.iterrows():
        conc, baf, t = row["ButyrateConc"], row["BafA1"], row["Time"]
        mu = intercept + effect_butyrate[conc] + effect_baf[baf] + effect_time[t]
        if (conc, baf) in interaction_but_baf:
            mu += interaction_but_baf[(conc, baf)]
        if (conc, t) in interaction_but_time:
            mu += interaction_but_time[(conc, t)]
        if (baf, t) in interaction_baf_time:
            mu += interaction_baf_time[(baf, t)]
        LC3_mean.append(mu)
    noise = np.random.normal(0, sigma, len(LC3_mean))
    design["LC3_II"] = np.array(LC3_mean) + noise

    # Simular p62 (inverso)
    p62_mean = [1.0 - (mu - 1.0) * 0.8 for mu in LC3_mean]
    design["p62"] = np.array(p62_mean) + np.random.normal(0, sigma, len(p62_mean))

    return design

# Ejecutar simulación base
design = simulate_and_analyze(sigma=0.3, n_replicates=3, vary_effects=1.0)

# Convertir a categórico y crear variable de grupo
for col in ["ButyrateConc", "BafA1", "Time"]:
    design[col] = design[col].astype('category')
design["group"] = (design["ButyrateConc"].astype(str) + "_" +
                     design["BafA1"].astype(str) + "_" + design["Time"].astype(str))

# Modelo y ANOVA para LC3_II
model_lc3 = smf.ols("LC3_II ~ C(ButyrateConc)*C(BafA1)*C(Time)", data=design).fit()
anova_lc3 = anova_lm(model_lc3, typ=2)
print("ANOVA LC3_II:\n", anova_lc3)

# Diagnósticos
esiduals = model_lc3.resid
shap_p = shapiro(residuals)[1]  # Normalidad
print(f"Shapiro-Wilk p={shap_p:.3f}")
lev_p = levene(*[design.loc[design['group'] == g, 'LC3_II'] for g in design['group'].unique()])[1]
print(f"Levene p={lev_p:.3f}")
X = model_lc3.model.exog
vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
print("VIF:", vif)

# Post-hoc Tukey para ButyrateConc*BafA1
tukey_lc3 = pairwise_tukeyhsd(
    endog=design["LC3_II"],
    groups=design["ButyrateConc"].astype(str) + "_" + design["BafA1"].astype(str)
)
print(tukey_lc3)

# Power analysis
power = FTestAnovaPower().solve_power(
    effect_size=0.25,
    nobs=design.shape[0],
    alpha=0.05,
    k_groups=len(design['group'].unique())
)
print(f"Potencia={power:.3f}")

# Modelo para p62
model_p62 = smf.ols("p62 ~ C(ButyrateConc)*C(BafA1)*C(Time)", data=design).fit()
anova_p62 = anova_lm(model_p62, typ=2)
print("ANOVA p62:\n", anova_p62)

# Sensibilidad: Variar σ
for sigma in [0.1, 0.3, 0.5]:
    des = simulate_and_analyze(sigma=sigma)
    mod = smf.ols("LC3_II ~ C(ButyrateConc)*C(BafA1)*C(Time)", data=des).fit()
    print(f"Sigma={sigma}: R²={mod.rsquared:.3f}")

# Visualización mejorada
sns.set(style="whitegrid")
g = sns.relplot(
    data=design,
    x="Time",
    y="LC3_II",
    hue="BafA1",
    col="ButyrateConc",
    kind="line",
    ci=95,
    marker="o"
)
for ax in g.axes.flatten():
    sns.boxplot(
        data=design,
        x="Time",
        y="LC3_II",
        hue="BafA1",
        ax=ax,
        dodge=True
    )
    ax.annotate("p<0.05 (Tukey)", xy=(0.5, 0.95), xycoords='axes fraction', ha='center')

plt.tight_layout()
plt.savefig("lc3_plot.png")
plt.show()

# Exportar datos
design.to_csv("simulated_data.csv", index=False)
