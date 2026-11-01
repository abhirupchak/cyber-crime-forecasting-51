"""Problem 51: forecasting + profiling state cyber-crime (NCRB 2018-2022).
Train 2018-2021, forecast 2022. Pooled trend models + clustering.
Regenerates every number/figure. Seed 42.
"""
import pandas as pd, numpy as np, os
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs"); os.makedirs(OUT, exist_ok=True)
XLS = os.path.join(BASE, "data", "datafile.xls")
rng = np.random.seed(42)

df = pd.read_excel(XLS, engine="calamine", header=None)
df.columns = ["sl","state","2018","2019","2020","2021","2022"]
df = df.iloc[1:].copy()
df = df[~df["state"].isin(["Total States","Total UT","Total (All India)"])].reset_index(drop=True)
for y in ["2018","2019","2020","2021","2022"]:
    df[y] = pd.to_numeric(df[y], errors="coerce")
df.to_csv(os.path.join(OUT,"clean_states.csv"), index=False)
# supervised set: exclude Ladakh (only 2020-2022)
dfm = df[df.state!="Ladakh"].copy().reset_index(drop=True)  # 35
YEARS=[2018,2019,2020,2021,2022]
long=[]
for _,r in dfm.iterrows():
    for y in YEARS:
        long.append((r["state"],y,y-2018,float(r[str(y)])))
long=pd.DataFrame(long,columns=["state","year","t","cases"])
long["logcases"]=np.log1p(long["cases"])
train=long[long.year<=2021].copy(); test=long[long.year==2022].copy()
enc=OneHotEncoder(sparse_output=False,handle_unknown="ignore")
S_tr=enc.fit_transform(train[["state"]]); S_te=enc.transform(test[["state"]])
t_tr=train["t"].values.reshape(-1,1); t_te=test["t"].values.reshape(-1,1)
y_tr_log=train["logcases"].values; y_te=test["cases"].values; y_te_log=test["logcases"].values
back=lambda a: np.expm1(np.maximum(a,0))
def m(yt,yp):
    return float(np.sqrt(mean_squared_error(yt,yp))),float(mean_absolute_error(yt,yp)),float(r2_score(yt,yp))
rows=[]
pers=dfm["2021"].values.astype(float)
rmse,mae,r2=m(y_te,pers); rows.append(("Persistence (2022=2021)",rmse,mae,r2))
mn=dfm[["2018","2019","2020","2021"]].mean(axis=1).values
rmse0,mae0,r20=m(y_te,mn); rows.append(("State mean 2018-21",rmse0,mae0,r20))
Xtr=np.hstack([S_tr,t_tr]); Xte=np.hstack([S_te,t_te])
lr=LinearRegression().fit(Xtr,y_tr_log); pl=back(lr.predict(Xte))
rmse1,mae1,r21=m(y_te,pl); rows.append(("Pooled OLS (state FE + t)",rmse1,mae1,r21))
print("OLS slope(log/yr)=%.4f"%lr.coef_[-1])
# ridge alpha by 2021 holdout
def fit_ridge(Xtr_,y_,a): return Ridge(alpha=a).fit(Xtr_,y_)
best=None
tr20=train[train.year<=2020]; tr21=train[train.year==2021]
E20=enc.transform(tr20[["state"]]); E21=enc.transform(tr21[["state"]])
T20=tr20["t"].values.reshape(-1,1); T21=tr21["t"].values.reshape(-1,1)
y20=np.log1p(tr20["cases"].values); y21=tr21["cases"].values
for a in [0.1,1.0,10.0,100.0,1000.0]:
    md=Ridge(alpha=a).fit(np.hstack([E20,T20]),y20)
    pv=back(md.predict(np.hstack([E21,T21])))
    v=np.sqrt(mean_squared_error(y21,pv))
    if best is None or v<best[1]: best=(a,v)
print("ridge alpha:",best)
rg=Ridge(alpha=best[0]).fit(Xtr,y_tr_log); pr=back(rg.predict(Xte))
rmse2,mae2,r22=m(y_te,pr); rows.append((f"Ridge a={best[0]} (state FE + t)",rmse2,mae2,r22))
# poly2
pf=PolynomialFeatures(2,include_bias=False)
P20=pf.fit_transform(T20); P21=pf.transform(T21)
Ptr=pf.fit_transform(t_tr); Pte=pf.transform(t_te)
best2=None
for a in [0.1,1.0,10.0,100.0,1000.0]:
    md=Ridge(alpha=a).fit(np.hstack([E20,P20]),y20)
    pv=back(md.predict(np.hstack([E21,P21])))
    v=np.sqrt(mean_squared_error(y21,pv))
    if best2 is None or v<best2[1]: best2=(a,v)
print("poly alpha:",best2)
po=Ridge(alpha=best2[0]).fit(np.hstack([S_tr,Ptr]),y_tr_log); pp=back(po.predict(np.hstack([S_te,Pte])))
rmse3,mae3,r23=m(y_te,pp); rows.append((f"Poly-2 + Ridge a={best2[0]}",rmse3,mae3,r23))
res=pd.DataFrame(rows,columns=["Model","RMSE","MAE","R2"])
res.to_csv(os.path.join(OUT,"table_regression.csv"),index=False)
print(res.to_string(index=False))
per=pd.DataFrame({"state":dfm["state"],"actual2022":y_te.astype(int),"persist":pers.astype(int),
 "ols":np.round(pl).astype(int),"ridge":np.round(pr).astype(int),"poly":np.round(pp).astype(int),
 "abs_err_persist":np.abs(y_te-pers).astype(int),"abs_err_ridge":np.abs(y_te-pr).astype(int)})
per.to_csv(os.path.join(OUT,"per_state_forecast.csv"),index=False)
# ---------------- rolling-origin backtest: 2020, 2021, 2022 ----------------
def _bt_design(tr_, te_):
    e = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    return e.fit_transform(tr_[["state"]]), e.transform(te_[["state"]])
bt_rows = []
for ty in [2020, 2021, 2022]:
    tr_ = long[long.year < ty]; te_ = long[long.year == ty]
    Etr, Ete = _bt_design(tr_, te_)
    ttr = tr_["t"].values.reshape(-1, 1); tte = te_["t"].values.reshape(-1, 1)
    yv = te_["cases"].values
    prev = long[long.year == ty-1].set_index("state").loc[te_["state"], "cases"].values
    sm = tr_.groupby("state")["cases"].mean().loc[te_["state"]].values
    ols = back(LinearRegression().fit(np.hstack([Etr, ttr]), tr_["logcases"]).predict(np.hstack([Ete, tte])))
    rg = back(Ridge(alpha=0.1).fit(np.hstack([Etr, ttr]), tr_["logcases"]).predict(np.hstack([Ete, tte])))
    preds = {"Persistence": prev, "State mean": sm, "Pooled OLS": ols, "Ridge a=0.1": rg}
    if tr_.year.nunique() >= 3:
        t2tr = np.hstack([ttr, ttr**2]); t2te = np.hstack([tte, tte**2])
        preds["Poly-2 + Ridge a=1.0"] = back(Ridge(alpha=1.0).fit(np.hstack([Etr, t2tr]), tr_["logcases"]).predict(np.hstack([Ete, t2te])))
    for name, yh in preds.items():
        rm, ma, r2 = m(yv, yh)
        bt_rows.append((ty, name, rm, ma, r2, float(np.median(np.abs(yv-yh)))))
bt = pd.DataFrame(bt_rows, columns=["test_year", "Model", "RMSE", "MAE", "R2", "median_AE"])
bt.to_csv(os.path.join(OUT, "backtest_folds.csv"), index=False)
piv = bt.pivot(index="test_year", columns="Model", values="MAE")
mase = (piv.div(piv["Persistence"], axis=0)).mean().rename("MASE").reset_index()
summ = (bt.groupby("Model").agg(folds=("test_year", "count"), mean_RMSE=("RMSE", "mean"),
        mean_MAE=("MAE", "mean"), mean_R2=("R2", "mean"), mean_median_AE=("median_AE", "mean"))
        .reset_index().merge(mase, on="Model", how="left"))
summ.to_csv(os.path.join(OUT, "backtest_summary.csv"), index=False)
print(bt.to_string(index=False)); print(summ.to_string(index=False))
# robust view of the 2022 holdout
te22 = long[long.year == 2022]; y22 = te22["cases"].values
Etr, Ete = _bt_design(long[long.year <= 2021], te22)
t1 = long[long.year <= 2021]["t"].values.reshape(-1, 1); t2 = te22["t"].values.reshape(-1, 1)
p22 = long[long.year == 2021].set_index("state").loc[te22["state"], "cases"].values
o22 = back(LinearRegression().fit(np.hstack([Etr, t1]), long[long.year <= 2021]["logcases"]).predict(np.hstack([Ete, t2])))
r22 = back(Ridge(alpha=0.1).fit(np.hstack([Etr, t1]), long[long.year <= 2021]["logcases"]).predict(np.hstack([Ete, t2])))
no_tel = (te22["state"] != "Telangana").values
top10 = long[long.year == 2021].nlargest(10, "cases")["state"]
m10 = te22["state"].isin(top10).values
rob = pd.DataFrame([
    ["All 35 entities", "Persistence", m(y22, p22)[0], float(np.median(np.abs(y22-p22)))],
    ["All 35 entities", "Ridge", m(y22, r22)[0], float(np.median(np.abs(y22-r22)))],
    ["Excluding Telangana", "Persistence", m(y22[no_tel], p22[no_tel])[0], float(np.median(np.abs(y22-p22)[no_tel]))],
    ["Excluding Telangana", "Ridge", m(y22[no_tel], r22[no_tel])[0], float(np.median(np.abs(y22-r22)[no_tel]))],
    ["Top-10 volume States", "Persistence", m(y22[m10], p22[m10])[0], float(np.median(np.abs(y22-p22)[m10]))],
    ["Top-10 volume States", "Ridge", m(y22[m10], r22[m10])[0], float(np.median(np.abs(y22-r22)[m10]))],
], columns=["subset", "Model", "RMSE", "median_AE"])
rob.to_csv(os.path.join(OUT, "backtest_robustness.csv"), index=False)
print(rob.to_string(index=False))

# All-India penetration correlation (cited aggregates, Dec-end; UPI FY volumes bn)
ai=pd.DataFrame({"year":[2018,2019,2020,2021,2022],
 "cyber":[27248,44735,50035,52974,65893],
 "teledensity":[91.45,88.56,86.38,85.91,84.56],
 "internet_mn":[604.21,718.74,795.18,829.30,865.90],
 "upi_bn":[0.92,5.34,12.52,22.33,45.96]})
ai.to_csv(os.path.join(OUT,"allindia_penetration.csv"),index=False)
for c in ["internet_mn","upi_bn","teledensity"]:
    r=np.corrcoef(np.log(ai["cyber"]),np.log(ai[c].clip(lower=0.1)) if c!="teledensity" else ai[c])[0,1]
    print(f"log-cyber vs {c}: r={r:.3f}")
print("cyber CAGR 2018-22: %.1f%%"%(((65893/27248)**(1/4)-1)*100))
# Clustering on log profiles
M=np.log1p(dfm[["2018","2019","2020","2021","2022"]].values.astype(float))
Z=StandardScaler().fit_transform(M)
sil=[]
for k in [2,3,4,5,6]:
    km=KMeans(n_clusters=k,n_init=20,random_state=42).fit(Z)
    ag=AgglomerativeClustering(n_clusters=k,linkage="ward").fit(Z)
    sil.append((k,float(silhouette_score(Z,km.labels_)),float(silhouette_score(Z,ag.labels_)),float(km.inertia_)))
    print(f"k={k} km-sil={sil[-1][1]:.3f} ward-sil={sil[-1][2]:.3f} inertia={sil[-1][3]:.1f}")
pd.DataFrame(sil,columns=["k","km_sil","ward_sil","inertia"]).to_csv(os.path.join(OUT,"table_silhouette.csv"),index=False)
K=4
km=KMeans(n_clusters=K,n_init=20,random_state=42).fit(Z)
ag=AgglomerativeClustering(n_clusters=K,linkage="ward").fit(Z)
# order clusters by mean 2022 level for stable labels
order=np.argsort([M[km.labels_==i,4].mean() for i in range(K)])
remap={old:new for new,old in enumerate(order)}
kml=np.array([remap[v] for v in km.labels_])
cl=pd.DataFrame({"state":dfm["state"],"kmeans":kml,"ward":ag.labels_,
 "mean_log":M.mean(axis=1),"cagr_18_21":((dfm["2021"]/dfm[["2018"]].replace(0,np.nan)["2018"])**(1/3)-1)})
cl.to_csv(os.path.join(OUT,"clusters.csv"),index=False)
print(cl.sort_values("kmeans").to_string(index=False))
# ---------------- State-level tele-density merge ----------------
td = pd.read_csv(os.path.join(BASE, "data", "teledensity_circle.csv"))
mp = pd.read_csv(os.path.join(BASE, "data", "circle_state_map.csv"))[["state", "circle"]]
tdl = td.melt(id_vars="circle", var_name="year", value_name="teledensity")
tdl["year"] = tdl["year"].astype(int)
panel = (long.merge(mp, on="state", how="left")
             .merge(tdl, on=["circle", "year"], how="left"))
assert panel["teledensity"].notna().all(), "unmapped state/circle"
panel.to_csv(os.path.join(OUT, "panel_with_teledensity.csv"), index=False)
ptr = panel[panel.year <= 2021]; pte = panel[panel.year == 2022]
enc2 = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
S2tr = enc2.fit_transform(ptr[["state"]]); S2te = enc2.transform(pte[["state"]])
T2tr = ptr["t"].values.reshape(-1, 1); T2te = pte["t"].values.reshape(-1, 1)
D2tr = ptr["teledensity"].values.reshape(-1, 1); D2te = pte["teledensity"].values.reshape(-1, 1)
LD2tr = np.log(ptr["teledensity"].values).reshape(-1, 1); LD2te = np.log(pte["teledensity"].values).reshape(-1, 1)
y2tr = ptr["logcases"].values
ols_td = LinearRegression().fit(np.hstack([S2tr, T2tr, D2tr]), y2tr)
ptd = back(ols_td.predict(np.hstack([S2te, T2te, D2te])))
ols_ltd = LinearRegression().fit(np.hstack([S2tr, T2tr, LD2tr]), y2tr)
pltd = back(ols_ltd.predict(np.hstack([S2te, T2te, LD2te])))
rg_td = Ridge(alpha=0.1).fit(np.hstack([S2tr, T2tr, D2tr]), y2tr)
prtd = back(rg_td.predict(np.hstack([S2te, T2te, D2te])))
r1 = m(y_te, ptd); r2 = m(y_te, pltd); r3 = m(y_te, prtd)
tbl_td = pd.DataFrame({
    "Model": ["Persistence", "OLS + tele-density", "OLS + log tele-density", "Ridge + tele-density"],
    "RMSE": [res.loc[0, "RMSE"], r1[0], r2[0], r3[0]],
    "MAE": [res.loc[0, "MAE"], r1[1], r2[1], r3[1]],
    "R2": [res.loc[0, "R2"], r1[2], r2[2], r3[2]],
    "coef_teledensity": [np.nan, ols_td.coef_[-1], ols_ltd.coef_[-1], rg_td.coef_[-1]],
})
tbl_td.to_csv(os.path.join(OUT, "table_teledensity.csv"), index=False)
print(tbl_td.to_string(index=False))
pooled_r = np.corrcoef(panel["logcases"], panel["teledensity"])[0, 1]
dem = panel.copy()
dem["lc"] = dem["logcases"] - dem.groupby("state")["logcases"].transform("mean")
dem["td"] = dem["teledensity"] - dem.groupby("state")["teledensity"].transform("mean")
within_r = np.corrcoef(dem["lc"], dem["td"])[0, 1]
x22 = panel[panel.year == 2022]
cross_r = np.corrcoef(x22["logcases"], x22["teledensity"])[0, 1]
print(f"tele-density r: pooled={pooled_r:.3f} within-state={within_r:.3f} cross-section2022={cross_r:.3f}")

# Figures: two-panel penetration chart (clear dual axes, explicit legends)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
x = np.array(YEARS)
ax1 = axes[0]
l1, = ax1.plot(x, ai["cyber"]/1000, "o-", color="#1f3a5f", label="Cyber cases (thousands)")
ax1.set_xlabel("Year"); ax1.set_ylabel("Cyber cases (thousands)", color="#1f3a5f")
ax1.set_xticks(x); ax1.tick_params(axis="y", labelcolor="#1f3a5f")
ax1b = ax1.twinx()
l2, = ax1b.plot(x, ai["internet_mn"], "s--", color="#e0a458", label="Internet subscribers (millions)")
ax1b.set_ylabel("Internet subscribers (millions)", color="#b07a2a")
ax1b.tick_params(axis="y", labelcolor="#b07a2a")
ax1.legend(handles=[l1, l2], loc="upper left", fontsize=7, frameon=False)
ax1.set_title("Cases and internet subscribers", fontsize=9)
ax2 = axes[1]
l3, = ax2.plot(x, ai["cyber"]/1000, "o-", color="#1f3a5f", label="Cyber cases (thousands)")
ax2.set_xlabel("Year"); ax2.set_ylabel("Cyber cases (thousands)", color="#1f3a5f")
ax2.set_xticks(x); ax2.tick_params(axis="y", labelcolor="#1f3a5f")
ax2b = ax2.twinx()
l4, = ax2b.plot(x, ai["upi_bn"], "^:", color="#2e7d5b", label="UPI volume (bn transactions)")
ax2b.set_ylabel("UPI volume (bn transactions)", color="#2e7d5b")
ax2b.tick_params(axis="y", labelcolor="#2e7d5b")
ax2.legend(handles=[l3, l4], loc="upper left", fontsize=7, frameon=False)
ax2.set_title("Cases and UPI volume", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT,"fig1_allindia.png"),dpi=150,bbox_inches="tight");plt.close()
plt.figure(figsize=(5,5))
mx=y_te.max()*1.05
plt.scatter(y_te,pr,s=18);plt.plot([0,mx],[0,mx],"r-")
plt.xlabel("Observed 2022 cases");plt.ylabel("Ridge predicted 2022")
plt.title("Ridge (state FE + t): observed vs predicted")
plt.savefig(os.path.join(OUT,"fig2_obs_pred.png"),dpi=150,bbox_inches="tight");plt.close()
plt.figure(figsize=(7,4))
for k in range(K):
    for idx in np.where(kml==k)[0]:
        plt.plot(YEARS,M[idx],color=f"C{k}",alpha=0.5)
    plt.plot(YEARS,M[kml==k].mean(axis=0),color=f"C{k}",lw=2.5,label=f"C{k} n={(kml==k).sum()}")
plt.xlabel("Year");plt.ylabel("log(1+cases)");plt.legend();plt.title("Growth profiles by k-means cluster (k=4)")
plt.xticks(YEARS)
plt.savefig(os.path.join(OUT,"fig3_clusters.png"),dpi=150,bbox_inches="tight");plt.close()
# surprise chart: actual - ridge prediction
sp=pd.DataFrame({"state":dfm["state"],"surprise":y_te-pr}).sort_values("surprise")
plt.figure(figsize=(6.5,6))
cols=["#c0392b" if v<0 else "#27ae60" for v in sp["surprise"]]
plt.barh(sp["state"],sp["surprise"],color=cols)
plt.axvline(0,color="k",lw=.8)
plt.xlabel("Actual 2022 cases minus ridge forecast")
plt.title("Where the forecast was wrong")
plt.tick_params(axis="y",labelsize=8)
plt.xticks(fontsize=8); plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT,"fig5_surprise.png"),dpi=150,bbox_inches="tight");plt.close()
# state-level tele-density scatter
plt.figure(figsize=(6.4,4.2))
plt.scatter(x22["teledensity"], x22["logcases"], s=28, color="#2c5282", alpha=.85)
for i, (_, r) in enumerate(x22.nlargest(5, "cases").iterrows()):
    off = (5, 5) if i % 2 == 0 else (5, -10)
    plt.annotate(r["state"], (r["teledensity"], r["logcases"]), fontsize=7,
                 xytext=off, textcoords="offset points")
plt.xlabel("Tele-density, December 2022 (%)")
plt.ylabel("log(1 + cases), 2022")
plt.title(f"State-level cross-section (r = {cross_r:.2f})")
plt.tight_layout()
plt.savefig(os.path.join(OUT,"fig6_teledensity.png"),dpi=150,bbox_inches="tight");plt.close()
L=linkage(Z,method="ward")
plt.figure(figsize=(9,4));dendrogram(L,labels=list(dfm["state"]),leaf_rotation=90,leaf_font_size=7)
plt.title("Ward dendrogram of state growth profiles");plt.tight_layout()
plt.savefig(os.path.join(OUT,"fig4_dendrogram.png"),dpi=150,bbox_inches="tight");plt.close()
print("done ->",OUT)
