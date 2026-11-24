"""
P2 - NLP on incidents and audit actions
---------------------------------------
Business question: across hundreds of free-text incident / audit records, what are
the recurring themes and root causes, and which are growing?

DATA: synthetic free-text records (templated + noise) generated here with a fixed
seed. Each record is given a hidden "true" theme ONLY so we can check the unsupervised
model; a real dataset would not have this label.
Method: TF-IDF -> NMF topic model -> theme labels -> trend + overdue-action analysis.
"""
import numpy as np, pandas as pd, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score

rng = np.random.default_rng(7)
OUT = "."

# ------------------------------------------------------------ 1. synthetic records
# each theme has a pool of shared vocabulary; sentences are composed from random picks,
# so wording varies like real free text while topics remain statistically separable
themes = {
 "Third-party / carrier failure": dict(
    subj=["courier partner", "third party carrier", "contracted haulier", "supplier", "logistics provider", "delivery partner"],
    event=["missed collection", "failed to deliver", "capacity shortfall", "late delivery", "system outage", "failed to scan parcels"],
    cause=["causing delayed customer deliveries", "leaving stock shortages on promoted lines", "disrupting store replenishment", "with no contingency carrier available", "during peak trading"]),
 "System / IT outage": dict(
    subj=["website checkout", "payment gateway", "stock system", "order management platform", "mobile app", "database"],
    event=["unavailable", "timeout", "sync failure", "latency issue", "failed deployment", "service degradation"],
    cause=["causing failed customer orders", "showing incorrect availability online", "after release change", "slowing click and collect processing", "with rollback delayed"]),
 "Process / control failure": dict(
    subj=["manual stock count", "price change", "refund authorisation", "leaver access removal", "supplier onboarding check", "approval step"],
    event=["not completed", "bypassed", "missing evidence", "not approved", "skipped", "performed late"],
    cause=["leading to margin error", "with no secondary check", "so control did not operate", "found during audit testing", "creating financial loss"]),
 "Colleague availability / capability": dict(
    subj=["warehouse team", "new starters", "shift cover", "key person", "contact centre team", "picking colleagues"],
    event=["short notice absence", "not trained", "high attrition", "below minimum staffing", "single person dependency", "overtime limit reached"],
    cause=["reducing throughput on peak shift", "causing picking errors", "leaving no cover for critical task", "increasing delivery delays", "affecting colleague wellbeing"]),
 "Health, safety and product": dict(
    subj=["warehouse aisle", "racking", "fire door", "product recall notice", "manual handling task", "vehicle yard"],
    event=["near miss", "damage identified", "obstructed", "not actioned", "injury reported", "failed inspection"],
    cause=["escalated to safety lead", "during store walk", "beyond required timeframe", "with corrective repair needed", "risk of colleague harm"]),
 "Cyber / data": dict(
    subj=["phishing email", "supplier portal account", "customer data export", "unpatched server", "privileged account", "laptop"],
    event=["clicked by colleague", "unauthorised access attempt", "sent to wrong recipient", "vulnerability flagged", "credential compromise", "lost device"],
    cause=["reported to security team", "privacy review opened", "beyond agreed patching deadline", "with possible data exposure", "detected by monitoring"]),
}
def make_text(theme, loc):
    d = themes[theme]
    return f"{rng.choice(d['subj'])} {rng.choice(d['event'])} {rng.choice(d['cause'])} at {loc}"
locs = ["Daventry", "Basingstoke", "Bristol", "Manchester", "Glasgow", "Leeds", "Peterborough", "Store 214", "Store 087"]
names = list(themes)
# third-party and cyber are made to GROW over time (planted trend to detect)
quarters = pd.period_range("2024Q1", "2025Q4", freq="Q")
rows = []
for qi, q in enumerate(quarters):
    base = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    base[0] += 0.18 * qi; base[5] += 0.15 * qi
    p = base / base.sum()
    for _ in range(rng.integers(55, 70)):
        t = rng.choice(len(names), p=p)
        txt = make_text(names[t], rng.choice(locs))
        if rng.random() < .3: txt += " " + rng.choice(["escalated to area manager", "customer impact low", "raised via incident log", "reviewed by control owner", "action agreed with owner"])
        due_days = int(rng.integers(20, 90))
        closed = rng.random() < (0.85 - (0.18 if t in (0, 5) else 0) - 0.02 * qi * (t in (0, 5)))
        age = int(rng.integers(5, due_days + 10) if closed else rng.integers(5, 140))
        rows.append(dict(quarter=str(q), text=txt, true_theme=names[t], status="Closed" if closed else "Open",
                         days_to_close_or_open=age, due_days=due_days))
df = pd.DataFrame(rows)
df["overdue"] = (df.days_to_close_or_open > df.due_days)
df.to_csv(f"{OUT}/synthetic_incidents.csv", index=False)

# ------------------------------------------------------------ 2. topic model
STOP = list(__import__("sklearn.feature_extraction.text", fromlist=["x"]).ENGLISH_STOP_WORDS) + \
       [l.lower() for l in " ".join(locs).split()] + ["escalated", "manager", "customer", "impact", "low", "raised", "log",
        "reviewed", "owner", "agreed", "action", "area", "control", "store"]
vec = TfidfVectorizer(stop_words=STOP, ngram_range=(1, 2), min_df=3, max_df=0.5, token_pattern=r"[a-zA-Z]{3,}")
X = vec.fit_transform(df.text)
K = 6
nmf = NMF(n_components=K, random_state=0, init="nndsvda", max_iter=600)
W = nmf.fit_transform(X)
df["topic"] = W.argmax(1)
terms = np.array(vec.get_feature_names_out())
top_terms = {k: ", ".join(terms[nmf.components_[k].argsort()[::-1][:6]]) for k in range(K)}

# map each discovered topic to the analyst-readable label by majority vote (for validation only)
ct = pd.crosstab(df.topic, df.true_theme)
label_map = ct.idxmax(axis=1).to_dict()
df["theme"] = df.topic.map(label_map)
ari = adjusted_rand_score(df.true_theme, df.topic)
purity = ct.max(axis=1).sum() / len(df)

# ------------------------------------------------------------ 3. analysis
by_q = pd.crosstab(df.quarter, df.theme)
share = by_q.div(by_q.sum(axis=1), axis=0) * 100
first, last = share.iloc[:2].mean(), share.iloc[-2:].mean()
trend = (last - first).sort_values(ascending=False).round(1)

open_ov = df[(df.status == "Open")].groupby("theme").agg(open_actions=("text", "size"), overdue=("overdue", "sum"))
open_ov["overdue_pct"] = (open_ov.overdue / open_ov.open_actions * 100).round(0)

terms_by_theme = pd.Series({k: v for k, v in top_terms.items()}).groupby(lambda k: label_map[k]).agg(" | ".join)
summary = pd.DataFrame({"top_terms": terms_by_theme,
                        "share_first_2q_%": first.round(1), "share_last_2q_%": last.round(1), "change_pts": trend}).join(open_ov)
summary.sort_values("change_pts", ascending=False).to_csv(f"{OUT}/theme_summary.csv")
df.drop(columns=["topic"]).to_csv(f"{OUT}/classified_incidents.csv", index=False)

# ------------------------------------------------------------ 4. charts
fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
share.plot.area(ax=ax[0], colormap="tab20c", alpha=.9, legend=False)
ax[0].set_title("Theme share of incidents by quarter (%)"); ax[0].set_ylabel("%"); ax[0].tick_params(axis="x", rotation=30)
ax[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.18), ncol=2)
trend.sort_values().plot.barh(ax=ax[1], color=["#c62828" if v > 2 else "#607d8b" for v in trend.sort_values()])
ax[1].set_title("Change in share: first 2 vs last 2 quarters (pts)")
ov = open_ov.sort_values("overdue_pct")
ax[2].barh(ov.index, ov.overdue_pct, color="#e07b00"); ax[2].set_title("% of OPEN actions already overdue")
for i, (v, n_) in enumerate(zip(ov.overdue_pct, ov.open_actions)): ax[2].text(v + 1, i, f"{int(v)}% (n={n_})", va="center", fontsize=8)
ax[2].set_xlim(0, 115)
plt.tight_layout(); plt.savefig(f"{OUT}/incident_themes.png", dpi=150); plt.close()

print(f"Records: {len(df)} | topics: {K}")
print(f"Validation vs hidden label: purity {purity:.0%}, adjusted Rand {ari:.2f}")
print("\nTopic top terms:"); [print(f"  {label_map[k]:38s} {v}") for k, v in top_terms.items()]
print("\nTrend (share change, pts):\n", trend.to_string())
print("\nOpen/overdue:\n", open_ov.sort_values("overdue_pct", ascending=False).to_string())
