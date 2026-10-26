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
# Figures
plt.figure(figsize=(7,4))
x=np.array(YEARS)
plt.plot(x,ai["cyber"]/1000,"o-",label="Cyber cases (000)")
plt.twinx().plot(x,ai["internet_mn"],"s--",color="tab:orange",label="Internet subs (M)")
plt.plot(x,np.array(ai["upi_bn"])*10,"^:",color="tab:green",label="UPI vol x10 (bn)")
plt.xlabel("Year");plt.title("All-India cyber cases vs digital penetration")
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
plt.savefig(os.path.join(OUT,"fig3_clusters.png"),dpi=150,bbox_inches="tight");plt.close()
# surprise chart: actual - ridge prediction
sp=pd.DataFrame({"state":dfm["state"],"surprise":y_te-pr}).sort_values("surprise")
plt.figure(figsize=(6.5,6))
cols=["#c0392b" if v<0 else "#27ae60" for v in sp["surprise"]]
plt.barh(sp["state"],sp["surprise"],color=cols)
plt.axvline(0,color="k",lw=.8)
plt.xlabel("Actual 2022 cases minus ridge forecast")
plt.title("Where the forecast was wrong")
plt.tick_params(axis="y",labelsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT,"fig5_surprise.png"),dpi=150,bbox_inches="tight");plt.close()
L=linkage(Z,method="ward")
plt.figure(figsize=(9,4));dendrogram(L,labels=list(dfm["state"]),leaf_rotation=90,leaf_font_size=7)
plt.title("Ward dendrogram of state growth profiles");plt.tight_layout()
plt.savefig(os.path.join(OUT,"fig4_dendrogram.png"),dpi=150,bbox_inches="tight");plt.close()
print("done ->",OUT)
