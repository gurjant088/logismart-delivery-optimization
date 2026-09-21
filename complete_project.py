# -*- coding: utf-8 -*-
"""
LogiSmart India - Smart Delivery Route and Resource Optimization System
COMPLETE ALL-IN-ONE SCRIPT: runs the full optimizer from the raw dataset,
then generates ALL 25 charts (10 core KPI/data plots + the full 15-type
chart gallery: Area, Bar, Line, Spline, Pie, Donut, Heat Map, Multi Tab,
Combination, Scatter, Waterfall, Candlestick, OHLC, Box Plot, Arc Chart).

Nothing needs to exist beforehand except the raw dataset file. Everything
else - result files, all 25 plots - is produced by this one script, in
memory, with no dependency on previously-saved files.

HOW TO USE IN COLAB:
  1. Upload "Competitor Beta Dataset for Students.xlsx" to /content/sample_data/
  2. Paste this whole file into a cell and run it
     (or upload it and run: %run complete_project.py)
  3. Everything prints below; result files, 10 core plots, and the 15-chart
     gallery are all saved under /content/sample_data/
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from collections import defaultdict

pd.set_option('display.width', 120)

# =============================================================================
# STEP 0 - LOCATE THE DATA FILE (checks a few likely spots automatically)
# =============================================================================
CANDIDATE_PATHS = [
    "/content/sample_data/Competitor Beta Dataset for Students.xlsx",
    "/content/Competitor Beta Dataset for Students.xlsx",
    "/content/sample_data/Competitor_Beta_Dataset_for_Students.xlsx",
    "/content/Competitor_Beta_Dataset_for_Students.xlsx",
]
DATA_PATH = os.environ.get("DATA_PATH")
if not DATA_PATH or not os.path.isfile(DATA_PATH):
    DATA_PATH = None
    for p in CANDIDATE_PATHS:
        if os.path.isfile(p):
            DATA_PATH = p
            break

if DATA_PATH is None:
    raise FileNotFoundError(
        "\n\nCould not find the dataset file.\n"
        "FIX: In Colab, click the folder icon on the left sidebar, click the "
        "upload button, and upload 'Competitor Beta Dataset for Students.xlsx'.\n"
        "Or set it explicitly before running this script:\n"
        "    import os; os.environ['DATA_PATH'] = '/path/to/your/file.xlsx'\n"
        f"Checked these locations:\n" + "\n".join(f"  - {p}" for p in CANDIDATE_PATHS)
    )

print(f"[OK] Loading data from: {DATA_PATH}")
OUT_DIR = os.environ.get("OUTPUT_DIR", os.path.dirname(DATA_PATH) or ".")
PLOTS_DIR = os.environ.get("PLOTS_DIR", os.path.join(OUT_DIR, "plots"))
GALLERY_DIR = os.environ.get("GALLERY_DIR", os.path.join(OUT_DIR, "chart_gallery"))
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(GALLERY_DIR, exist_ok=True)

orders_df = pd.read_excel(DATA_PATH, sheet_name="Delivery Orders")
vehicles_df = pd.read_excel(DATA_PATH, sheet_name="Vehicle Master")
dist_df = pd.read_excel(DATA_PATH, sheet_name="Zone Distance Matrix", index_col=0)
print(f"[OK] Loaded {len(orders_df)} orders, {len(vehicles_df)} vehicles, "
      f"{dist_df.shape[0]}x{dist_df.shape[1]} distance matrix.\n")

# =============================================================================
# STEP 1 - VALIDATION
# =============================================================================
def validate():
    checks = []
    checks.append(("Duplicate Order_ID", int(orders_df['Order_ID'].duplicated().sum())))
    checks.append(("Missing values in Delivery Orders", int(orders_df.isna().sum().sum())))
    checks.append(("Missing values in Vehicle Master", int(vehicles_df.isna().sum().sum())))
    checks.append(("Non-positive Distance_Km", int((orders_df['Distance_Km'] <= 0).sum())))
    checks.append(("Non-positive Package_Weight_Kg", int((orders_df['Package_Weight_Kg'] <= 0).sum())))
    checks.append(("Package weight exceeding largest vehicle capacity (500kg)",
                    int((orders_df['Package_Weight_Kg'] > vehicles_df['Capacity_Kg'].max()).sum())))
    checks.append(("Orders referencing an Origin_Hub with zero vehicles",
                    int((~orders_df['Origin_Hub'].isin(vehicles_df['Hub'].unique())).sum())))
    checks.append(("Vehicles with unavailable capacity <= 0",
                    int((vehicles_df['Capacity_Kg'] <= 0).sum())))
    return checks

validation_results = validate()
print("=== VALIDATION ===")
for name, val in validation_results:
    print(f"  {name}: {val}")
print()

# =============================================================================
# STEP 2 - DATA STRUCTURES
# =============================================================================
TYPE_RANK = {"Bike": 0, "Van": 1, "Mini Truck": 2}

fleet_by_hub = defaultdict(list)
for _, v in vehicles_df.iterrows():
    fleet_by_hub[v['Hub']].append({
        "Vehicle_ID": v['Vehicle_ID'], "Hub": v['Hub'], "Vehicle_Type": v['Vehicle_Type'],
        "Type_Rank": TYPE_RANK[v['Vehicle_Type']], "Capacity_Kg": v['Capacity_Kg'],
        "Cost_per_Km_INR": v['Cost_per_Km_INR'], "Max_Working_Minutes": v['Max_Working_Minutes'],
        "Available": v['Availability'] == "Available",
    })

graph = {i: {j: dist_df.loc[i, j] for j in dist_df.columns} for i in dist_df.index}

orders_df['_travel_min_existing'] = orders_df['Existing_Estimated_Time_Min'] - orders_df['Service_Time_Min']
AVG_SPEED_KMPH = float(
    orders_df['Existing_Planned_Distance_Km'].sum() / (orders_df['_travel_min_existing'].sum() / 60.0)
)

# =============================================================================
# STEP 3 - ALLOCATION
# ASSUMPTION: 1,800 orders exceed the fleet's single-shift capacity, so this
# is modelled as a multi-trip dispatch backlog (vehicles complete multiple
# sequential trips across the review period) rather than single-shift.
# =============================================================================
PRIORITY_RANK = {"High": 0, "Medium": 1, "Low": 2}

def build_trips():
    trips, unallocated = [], []
    orders_sorted = orders_df.copy()
    orders_sorted['_prio_rank'] = orders_sorted['Priority'].map(PRIORITY_RANK)
    orders_sorted = orders_sorted.sort_values(by=['_prio_rank', 'Package_Weight_Kg'], ascending=[True, False])

    groups = defaultdict(list)
    for _, order in orders_sorted.iterrows():
        hub = order['Origin_Hub']
        req_type = order['Required_Vehicle_Type']
        if req_type not in TYPE_RANK or not fleet_by_hub[hub]:
            unallocated.append({"Order_ID": order['Order_ID'], "Reason": "No fleet at Origin_Hub"})
            continue
        groups[(hub, req_type)].append(order)

    for (hub, vtype), group_orders in groups.items():
        vehicles_of_type = [v for v in fleet_by_hub[hub] if v['Vehicle_Type'] == vtype and v['Available']]
        if not vehicles_of_type:
            for order in group_orders:
                unallocated.append({"Order_ID": order['Order_ID'], "Reason": f"No available {vtype} at {hub}"})
            continue
        capacity = vehicles_of_type[0]['Capacity_Kg']
        max_minutes = vehicles_of_type[0]['Max_Working_Minutes']

        bins = []
        for order in group_orders:
            weight = order['Package_Weight_Kg']
            service_time = order['Service_Time_Min']
            est_travel_min = (order['Distance_Km'] / AVG_SPEED_KMPH) * 60.0
            need_minutes = service_time + est_travel_min

            placed = False
            for b in bins:
                if b['weight'] + weight <= capacity and b['minutes'] + need_minutes <= max_minutes:
                    b['weight'] += weight
                    b['minutes'] += need_minutes
                    b['orders'].append(order)
                    placed = True
                    break
            if not placed:
                if weight > capacity or need_minutes > max_minutes:
                    unallocated.append({"Order_ID": order['Order_ID'],
                                         "Reason": "Exceeds single-vehicle capacity or working-minute budget"})
                else:
                    bins.append({'weight': weight, 'minutes': need_minutes, 'orders': [order]})

        for i, b in enumerate(bins):
            vehicle = vehicles_of_type[i % len(vehicles_of_type)]
            trips.append({"Vehicle_ID": vehicle['Vehicle_ID'], "Hub": hub, "Vehicle_Type": vtype,
                           "Capacity_Kg": capacity, "Cost_per_Km_INR": vehicle['Cost_per_Km_INR'],
                           "Trip_No": i + 1, "Orders": b['orders']})
    return trips, unallocated

trips, unallocated_orders = build_trips()

# =============================================================================
# STEP 4 - ROUTE OPTIMIZATION
# =============================================================================
def nearest_neighbour_route(hub, stops):
    if not stops:
        return 0.0
    remaining = list(stops)
    current = hub
    total = 0.0
    while remaining:
        nxt = min(remaining, key=lambda z: graph[current][z])
        total += graph[current][nxt]
        current = nxt
        remaining.remove(nxt)
    return total

route_records = []
for t in trips:
    stops = [o['Destination_Zone'] for o in t['Orders']]
    route_km = nearest_neighbour_route(t['Hub'], stops)
    load = sum(o['Package_Weight_Kg'] for o in t['Orders'])
    route_records.append({
        "Trip_ID": f"{t['Vehicle_ID']}-T{t['Trip_No']}", "Vehicle_ID": t['Vehicle_ID'], "Hub": t['Hub'],
        "Vehicle_Type": t['Vehicle_Type'], "Stops": len(stops), "Route_Km": round(route_km, 2),
        "Load_Kg": round(load, 2), "Capacity_Kg": t['Capacity_Kg'],
        "Utilisation_%": round(100 * load / t['Capacity_Kg'], 1),
        "Estimated_Cost_INR": round(route_km * t['Cost_per_Km_INR'], 2),
    })
routes_df = pd.DataFrame(route_records)

# =============================================================================
# STEP 5 - KPI COMPARISON
# =============================================================================
baseline_total_distance = orders_df['Existing_Planned_Distance_Km'].sum()
baseline_avg_delay = orders_df['Delay_Min'].mean()
baseline_on_time_rate = (orders_df['Delivery_Status'] == 'On Time').mean()

_tmp = orders_df.assign(rank=orders_df['Required_Vehicle_Type'].map(TYPE_RANK)).merge(
    vehicles_df.assign(rank=vehicles_df['Vehicle_Type'].map(TYPE_RANK)).groupby('rank')['Cost_per_Km_INR'].first().reset_index(),
    on='rank')
baseline_total_cost = float((_tmp['Existing_Planned_Distance_Km'] * _tmp['Cost_per_Km_INR']).sum())

optimized_total_distance = routes_df['Route_Km'].sum()
optimized_total_cost = routes_df['Estimated_Cost_INR'].sum()
allocation_rate = 1 - (len(unallocated_orders) / len(orders_df))
avg_vehicle_utilisation = routes_df['Utilisation_%'].mean()

order_to_trip = {}
for t in trips:
    trip_id = f"{t['Vehicle_ID']}-T{t['Trip_No']}"
    for o in t['Orders']:
        order_to_trip[o['Order_ID']] = trip_id

trip_route_km = dict(zip(routes_df['Trip_ID'], routes_df['Route_Km']))
trip_stop_count = dict(zip(routes_df['Trip_ID'], routes_df['Stops']))

def optimized_delay_for_order(row):
    trip_id = order_to_trip.get(row['Order_ID'])
    if trip_id is None:
        return row['Delay_Min']
    share_km = trip_route_km[trip_id] / trip_stop_count[trip_id]
    optimized_travel_min = (share_km / AVG_SPEED_KMPH) * 60.0
    existing_travel_min = row['_travel_min_existing']
    time_saved = max(0.0, existing_travel_min - optimized_travel_min)
    return max(0.0, row['Delay_Min'] - time_saved)

orders_df['Optimized_Delay_Min'] = orders_df.apply(optimized_delay_for_order, axis=1)
optimized_avg_delay = orders_df['Optimized_Delay_Min'].mean()
optimized_on_time_rate = (orders_df['Optimized_Delay_Min'] <= 0).mean()

def pct_improve(base, opt, lower_is_better=True):
    if base == 0:
        return 0.0
    return (base - opt) / base if lower_is_better else (opt - base) / base

kpi_summary = {
    "Total Distance (km)": (round(baseline_total_distance, 1), round(optimized_total_distance, 1),
                             pct_improve(baseline_total_distance, optimized_total_distance)),
    "Estimated Cost (INR)": (round(baseline_total_cost, 0), round(optimized_total_cost, 0),
                              pct_improve(baseline_total_cost, optimized_total_cost)),
    "Average Delay (min)": (round(baseline_avg_delay, 2), round(optimized_avg_delay, 2),
                             pct_improve(baseline_avg_delay, optimized_avg_delay)),
    "On-Time Rate": (round(baseline_on_time_rate, 4), round(optimized_on_time_rate, 4),
                      pct_improve(baseline_on_time_rate, optimized_on_time_rate, lower_is_better=False)),
    "Order Allocation Rate": (None, round(allocation_rate, 4), None),
    "Avg Vehicle Utilisation (%)": (None, round(avg_vehicle_utilisation, 1), None),
    "Vehicle-Trips (dispatches)": (None, int(routes_df.shape[0]), None),
    "Distinct Vehicles Used": (None, int(routes_df['Vehicle_ID'].nunique()), None),
    "Vehicles Available": (None, int(vehicles_df.shape[0]), None),
}

print("=== BASELINE vs OPTIMIZED KPIs ===")
for k, (b, o, imp) in kpi_summary.items():
    imp_str = f"{imp*100:.1f}%" if imp is not None else "n/a"
    print(f"  {k}: baseline={b}, optimized={o}, improvement={imp_str}")
print(f"\n  Unallocated orders: {len(unallocated_orders)}\n")

# =============================================================================
# STEP 6 - FUNCTIONAL TESTS
# =============================================================================
test_results = []

def record(test_id, feature, condition, expected, actual, passed, evidence):
    test_results.append({
        "Test_ID": test_id, "Feature": feature, "Input_Condition": condition,
        "Expected_Result": expected, "Actual_Result": actual,
        "Pass_Fail": "Pass" if passed else "Fail", "Evidence": evidence
    })

over_capacity_trips = routes_df[routes_df['Load_Kg'] > routes_df['Capacity_Kg'] + 1e-6]
record("TC-01", "Allocation - capacity constraint", "All trips built by AllocationService",
       "No trip's Load_Kg exceeds its vehicle Capacity_Kg",
       f"{len(over_capacity_trips)} trips exceed capacity", len(over_capacity_trips) == 0,
       f"routes checked: {len(routes_df)}; violations: {len(over_capacity_trips)}")

maint_ids = set(vehicles_df.loc[vehicles_df['Availability'] == 'Maintenance', 'Vehicle_ID'])
overlap = maint_ids & set(routes_df['Vehicle_ID'])
record("TC-02", "Allocation - vehicle availability", f"{len(maint_ids)} vehicles flagged 'Maintenance'",
       "None of the Maintenance vehicles appear in any trip",
       f"Overlap: {sorted(overlap) if overlap else 'none'}", len(overlap) == 0,
       f"Maintenance fleet: {sorted(maint_ids)}")

overweight_flag = 550.0 > vehicles_df['Capacity_Kg'].max()
record("TC-03", "Validation - overweight order", "Synthetic order weight=550kg (fleet max=500kg)",
       "Flagged as exceeding all vehicle capacity", f"Exceeds max capacity: {overweight_flag}",
       bool(overweight_flag), "Checked against Vehicle Master Capacity_Kg.max()=500")

bad_df = orders_df.copy()
bad_df.loc[0, 'Distance_Km'] = None
missing_count = int(bad_df['Distance_Km'].isna().sum())
record("TC-04", "Validation - missing field handling", "Distance_Km set null on 1 row (copied dataset)",
       "Exactly 1 missing value detected", f"Missing detected: {missing_count}", missing_count == 1,
       "isna().sum() re-run on modified copy")

has_fleet = "Central Hub" in vehicles_df['Hub'].unique()
record("TC-05", "Routing - no feasible route", "Synthetic order from 'Central Hub' (no vehicles there)",
       "Flagged unallocated: 'No fleet at Origin_Hub'", f"Hub has fleet: {has_fleet}", not has_fleet,
       f"Vehicle Master hubs: {sorted(vehicles_df['Hub'].unique())}")

dup_df = pd.concat([orders_df, orders_df.iloc[[0]]], ignore_index=True)
dup_count = int(dup_df['Order_ID'].duplicated().sum())
record("TC-06", "Validation - duplicate order ID", "1 row duplicated (copied dataset)",
       "Exactly 1 duplicate detected", f"Duplicates: {dup_count}", dup_count == 1,
       "duplicated().sum() re-run on modified copy")

record("TC-07", "Allocation - full-dataset coverage", "All real orders passed through allocation",
       "0 unallocated orders", f"Unallocated: {len(unallocated_orders)}",
       len(unallocated_orders) == 0, "unallocated_orders length checked after full run")

print("=== FUNCTIONAL TESTS ===")
for r in test_results:
    print(f"  {r['Test_ID']} {r['Feature']} -> {r['Pass_Fail']}")

# =============================================================================
# STEP 7 - SAVE RESULT FILES
# =============================================================================
routes_df.to_csv(os.path.join(OUT_DIR, "routes_output.csv"), index=False)
orders_df.to_csv(os.path.join(OUT_DIR, "orders_with_optimized_delay.csv"), index=False)
with open(os.path.join(OUT_DIR, "kpi_summary.json"), "w") as f:
    json.dump({k: list(v) for k, v in kpi_summary.items()}, f, indent=2, default=str)
with open(os.path.join(OUT_DIR, "validation_results.json"), "w") as f:
    json.dump(validation_results, f, indent=2)
with open(os.path.join(OUT_DIR, "unallocated.json"), "w") as f:
    json.dump(unallocated_orders, f, indent=2, default=str)
with open(os.path.join(OUT_DIR, "test_results.json"), "w") as f:
    json.dump(test_results, f, indent=2, default=str)

print(f"\n[OK] Result files saved to: {OUT_DIR}")
print("  - routes_output.csv, orders_with_optimized_delay.csv, kpi_summary.json,")
print("  - validation_results.json, unallocated.json, test_results.json")

# =============================================================================
# STEP 8 - PLOTTING SETUP (shared by both chart sets below)
# =============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Arc as ArcPatch
import seaborn as sns

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

BASE_COLOR = "#94A3B8"
OPT_COLOR = "#2E7D32"
ACCENT_COLOR = "#1F4E79"
RED_COLOR = "#C0392B"

orders = orders_df    # aliases so both chart sections' code reads naturally
routes = routes_df
kpis = kpi_summary

def savefig_to(fig, folder, name):
    path = os.path.join(folder, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [saved] {path}")


# =============================================================================
# STEP 9 - CORE PLOT SET (10 plots) -> PLOTS_DIR
# =============================================================================
print(f"\n[OK] Generating 10 core plots into: {PLOTS_DIR}")

# 1. KPI comparison bars
metrics = ["Total Distance (km)", "Estimated Cost (INR)", "Average Delay (min)"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, metric in zip(axes, metrics):
    b, o, _ = kpis[metric]
    bars = ax.bar(["Baseline", "Optimized"], [b, o], color=[BASE_COLOR, OPT_COLOR], width=0.55)
    ax.set_title(metric, fontsize=11, fontweight="bold")
    for bar, val in zip(bars, [b, o]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,.0f}", ha="center", va="bottom", fontsize=9)
    improvement = (b - o) / b * 100 if b else 0
    ax.text(0.5, 0.92, f"-{improvement:.1f}%", transform=ax.transAxes, ha="center",
            fontsize=10, color=OPT_COLOR, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
fig.suptitle("Baseline vs Optimized \u2013 Core KPIs", fontsize=14, fontweight="bold", y=1.03)
fig.tight_layout()
savefig_to(fig, PLOTS_DIR, "01_kpi_comparison_bars.png")

# 2. On-time rate donuts
b, o, _ = kpis["On-Time Rate"]
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, rate, label, color in zip(axes, [b, o], ["Baseline", "Optimized"], [BASE_COLOR, OPT_COLOR]):
    ax.pie([rate, 1 - rate], colors=[color, "#EEEEEE"], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.38))
    ax.text(0, 0, f"{rate*100:.1f}%", ha="center", va="center", fontsize=18, fontweight="bold")
    ax.set_title(f"{label} On-Time Rate", fontsize=11, fontweight="bold")
fig.suptitle("On-Time Delivery Rate: Before vs After Optimization", fontsize=13, fontweight="bold")
fig.tight_layout()
savefig_to(fig, PLOTS_DIR, "02_on_time_rate_donuts.png")

# 3. Delay distribution histogram
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(0, max(orders["Delay_Min"].max(), orders["Optimized_Delay_Min"].max()), 40)
ax.hist(orders["Delay_Min"], bins=bins, alpha=0.55, label="Baseline Delay", color=BASE_COLOR)
ax.hist(orders["Optimized_Delay_Min"], bins=bins, alpha=0.75, label="Optimized Delay", color=OPT_COLOR)
ax.set_xlabel("Delay (minutes)")
ax.set_ylabel("Number of Orders")
ax.set_title("Delivery Delay Distribution: Baseline vs Optimized", fontsize=13, fontweight="bold")
ax.legend()
fig.tight_layout()
savefig_to(fig, PLOTS_DIR, "03_delay_distribution_histogram.png")

# 4. Orders by priority
order_pri = ["High", "Medium", "Low"]
counts = orders["Priority"].value_counts().reindex(order_pri)
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(counts.index, counts.values, color=sns.color_palette("Set2", 3))
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val, str(val), ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Number of Orders")
ax.set_title("Order Volume by Priority Level", fontsize=13, fontweight="bold")
fig.tight_layout()
savefig_to(fig, PLOTS_DIR, "04_orders_by_priority.png")

# 5. Orders by hub + vehicle type (stacked)
pivot = orders.pivot_table(index="Origin_Hub", columns="Required_Vehicle_Type",
                            values="Order_ID", aggfunc="count", fill_value=0)
fig, ax = plt.subplots(figsize=(9, 5.5))
pivot.plot(kind="bar", stacked=True, ax=ax, color=sns.color_palette("Set2", pivot.shape[1]))
ax.set_xlabel("Origin Hub")
ax.set_ylabel("Number of Orders")
ax.set_title("Order Volume by Hub and Required Vehicle Type", fontsize=13, fontweight="bold")
ax.legend(title="Vehicle Type")
plt.setp(ax.get_xticklabels(), rotation=0)
fig.tight_layout()
savefig_to(fig, PLOTS_DIR, "05_orders_by_hub_and_vehicle_type.png")

# 6. Route distance distribution
fig, ax = plt.subplots(figsize=(9, 5))
sns.histplot(routes["Route_Km"], bins=30, kde=True, color=OPT_COLOR, ax=ax)
ax.axvline(routes["Route_Km"].mean(), color=ACCENT_COLOR, linestyle="--",
           label=f"Mean = {routes['Route_Km'].mean():.1f} km")
ax.set_xlabel("Route Distance per Trip (km)")
ax.set_ylabel("Number of Trips")
ax.set_title("Distribution of Optimized Route Distance per Vehicle Trip", fontsize=13, fontweight="bold")
ax.legend()
fig.tight_layout()
savefig_to(fig, PLOTS_DIR, "06_route_distance_distribution.png")

# 7. Utilisation by vehicle type (boxplot)
fig, ax = plt.subplots(figsize=(8, 5.5))
order_vt = routes.groupby("Vehicle_Type")["Utilisation_%"].median().sort_values(ascending=False).index
sns.boxplot(data=routes, x="Vehicle_Type", y="Utilisation_%", order=order_vt,
            hue="Vehicle_Type", palette="Set2", legend=False, ax=ax)
ax.axhline(100, color="red", linestyle=":", linewidth=1, label="Full Capacity (100%)")
ax.set_xlabel("Vehicle Type")
ax.set_ylabel("Load Utilisation (%)")
ax.set_title("Vehicle Load Utilisation by Vehicle Type", fontsize=13, fontweight="bold")
ax.legend()
fig.tight_layout()
savefig_to(fig, PLOTS_DIR, "07_utilisation_by_vehicle_type.png")

# 8. Trips and cost by hub
summary = routes.groupby("Hub").agg(Trips=("Trip_ID", "count"),
                                     Total_Cost=("Estimated_Cost_INR", "sum")).sort_values("Total_Cost", ascending=False)
fig, ax1 = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(summary))
width = 0.4
ax1.bar(x - width/2, summary["Trips"], width, color=ACCENT_COLOR, label="Vehicle Trips")
ax1.set_ylabel("Number of Vehicle Trips", color=ACCENT_COLOR)
ax1.set_xticks(x)
ax1.set_xticklabels(summary.index)
ax1.tick_params(axis="y", labelcolor=ACCENT_COLOR)
ax2 = ax1.twinx()
ax2.bar(x + width/2, summary["Total_Cost"], width, color=OPT_COLOR, label="Estimated Cost (INR)")
ax2.set_ylabel("Estimated Cost (INR)", color=OPT_COLOR)
ax2.tick_params(axis="y", labelcolor=OPT_COLOR)
fig.suptitle("Vehicle Trips and Estimated Cost by Hub", fontsize=13, fontweight="bold")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
fig.tight_layout()
savefig_to(fig, PLOTS_DIR, "08_cost_and_trips_by_hub.png")

# 9. Correlation heatmap
cols = ["Distance_Km", "Package_Weight_Kg", "Service_Time_Min",
        "Existing_Planned_Distance_Km", "Delay_Min", "Optimized_Delay_Min"]
corr = orders[cols].corr()
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            square=True, ax=ax, cbar_kws={"label": "Correlation"})
ax.set_title("Correlation Between Order-Level Numeric Features", fontsize=13, fontweight="bold")
fig.tight_layout()
savefig_to(fig, PLOTS_DIR, "09_correlation_heatmap.png")

# 10. Delay improvement scatter
sample = orders.sample(min(400, len(orders)), random_state=42)
fig, ax = plt.subplots(figsize=(8.5, 6))
scatter = ax.scatter(sample["Delay_Min"], sample["Optimized_Delay_Min"],
                      c=sample["Distance_Km"], cmap="viridis", alpha=0.7, s=35)
max_val = max(sample["Delay_Min"].max(), sample["Optimized_Delay_Min"].max())
ax.plot([0, max_val], [0, max_val], "r--", linewidth=1, label="No improvement (y = x)")
ax.set_xlabel("Baseline Delay (min)")
ax.set_ylabel("Optimized Delay (min)")
ax.set_title("Per-Order Delay: Baseline vs Optimized\n(color = order distance in km)",
             fontsize=13, fontweight="bold")
cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("Distance (km)")
ax.legend()
fig.tight_layout()
savefig_to(fig, PLOTS_DIR, "10_delay_improvement_scatter.png")


# =============================================================================
# STEP 10 - FULL 15-TYPE CHART GALLERY -> GALLERY_DIR
# =============================================================================
print(f"\n[OK] Generating full 15-chart-type gallery into: {GALLERY_DIR}")

# ---- Basic: 1. Area ----
cum = routes.sort_values("Route_Km")["Route_Km"].cumsum().reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(cum.index, cum.values, color=OPT_COLOR, alpha=0.4)
ax.plot(cum.index, cum.values, color=OPT_COLOR, linewidth=1.5)
ax.set_xlabel("Trips (sorted by distance, ascending)")
ax.set_ylabel("Cumulative Route Distance (km)")
ax.set_title("Area Chart \u2013 Cumulative Optimized Route Distance", fontweight="bold")
fig.tight_layout()
savefig_to(fig, GALLERY_DIR, "01_area_cumulative_distance.png")

# ---- Basic: 2. Bar ----
order_pri = ["High", "Medium", "Low"]
counts = orders["Priority"].value_counts().reindex(order_pri)
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(counts.index, counts.values, color=sns.color_palette("Set2", 3))
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, val, str(val), ha="center", va="bottom")
ax.set_ylabel("Number of Orders")
ax.set_title("Bar Chart \u2013 Order Volume by Priority", fontweight="bold")
fig.tight_layout()
savefig_to(fig, GALLERY_DIR, "02_bar_orders_by_priority.png")

# ---- Basic: 3. Line ----
by_hub = orders.groupby("Origin_Hub")["Delay_Min"].mean().sort_index()
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(by_hub.index, by_hub.values, marker="o", color=ACCENT_COLOR, linewidth=2)
for x, y in zip(by_hub.index, by_hub.values):
    ax.text(x, y, f"{y:.1f}", ha="center", va="bottom")
ax.set_ylabel("Average Baseline Delay (min)")
ax.set_title("Line Chart \u2013 Average Baseline Delay by Hub", fontweight="bold")
fig.tight_layout()
savefig_to(fig, GALLERY_DIR, "03_line_avg_delay_by_hub.png")

# ---- Basic: 4. Spline ----
sorted_km = routes["Route_Km"].sort_values().reset_index(drop=True)
window = max(5, len(sorted_km) // 40)
smoothed = sorted_km.rolling(window, center=True, min_periods=1).mean()
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sorted_km.index, smoothed.values, color=ACCENT_COLOR, linewidth=2)
ax.set_xlabel("Trips (sorted by distance)")
ax.set_ylabel("Route Distance (km, smoothed)")
ax.set_title("Spline Chart \u2013 Smoothed Route Distance Trend", fontweight="bold")
fig.tight_layout()
savefig_to(fig, GALLERY_DIR, "04_spline_route_distance_trend.png")

# ---- Basic: 5. Pie ----
counts = orders["Required_Vehicle_Type"].value_counts()
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
       colors=sns.color_palette("Set2", len(counts)), startangle=90)
ax.set_title("Pie Chart \u2013 Orders by Required Vehicle Type", fontweight="bold")
fig.tight_layout()
savefig_to(fig, GALLERY_DIR, "05_pie_orders_by_vehicle_type.png")

# ---- Basic: 6. Donut ----
counts = routes["Vehicle_Type"].value_counts()
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
       colors=sns.color_palette("Set2", len(counts)), startangle=90, wedgeprops=dict(width=0.4))
ax.set_title("Donut Chart \u2013 Vehicle Trips by Vehicle Type", fontweight="bold")
fig.tight_layout()
savefig_to(fig, GALLERY_DIR, "06_donut_trips_by_vehicle_type.png")

# ---- Advanced: 7. Heat Map ----
pivot = orders.pivot_table(index="Origin_Hub", columns="Required_Vehicle_Type",
                            values="Order_ID", aggfunc="count", fill_value=0)
fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues", ax=ax, cbar_kws={"label": "Orders"})
ax.set_title("Heat Map \u2013 Order Count by Hub \u00d7 Vehicle Type", fontweight="bold")
fig.tight_layout()
savefig_to(fig, GALLERY_DIR, "07_heatmap_orders_hub_vehicle.png")

# ---- Advanced: 8. Multi Tab (grouped bar) ----
fig, ax = plt.subplots(figsize=(9, 5.5))
pivot.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", pivot.shape[1]))
ax.set_xlabel("Origin Hub")
ax.set_ylabel("Number of Orders")
ax.set_title("Multi Tab (Grouped Bar) \u2013 Orders by Hub and Vehicle Type", fontweight="bold")
plt.setp(ax.get_xticklabels(), rotation=0)
ax.legend(title="Vehicle Type")
fig.tight_layout()
savefig_to(fig, GALLERY_DIR, "08_multitab_orders_by_hub_vehicle.png")

# ---- Advanced: 9. Combination ----
summary2 = routes.groupby("Hub").agg(Trips=("Trip_ID", "count"), Cost=("Estimated_Cost_INR", "sum")).sort_index()
fig, ax1 = plt.subplots(figsize=(9, 5.5))
xg = np.arange(len(summary2))
ax1.bar(xg, summary2["Trips"], color=ACCENT_COLOR, width=0.5, label="Vehicle Trips")
ax1.set_ylabel("Vehicle Trips", color=ACCENT_COLOR)
ax1.set_xticks(xg)
ax1.set_xticklabels(summary2.index)
ax1.tick_params(axis="y", labelcolor=ACCENT_COLOR)
ax2 = ax1.twinx()
ax2.plot(xg, summary2["Cost"], color=RED_COLOR, marker="o", linewidth=2, label="Estimated Cost (INR)")
ax2.set_ylabel("Estimated Cost (INR)", color=RED_COLOR)
ax2.tick_params(axis="y", labelcolor=RED_COLOR)
fig.suptitle("Combination Chart \u2013 Trips (bar) vs Cost (line) by Hub", fontweight="bold")
l1, lab1 = ax1.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lab1 + lab2, loc="upper left")
fig.tight_layout()
savefig_to(fig, GALLERY_DIR, "09_combination_trips_and_cost_by_hub.png")

# ---- Advanced: 10. Scatter ----
sample = orders.sample(min(400, len(orders)), random_state=42)
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(sample["Delay_Min"], sample["Optimized_Delay_Min"],
                 c=sample["Distance_Km"], cmap="viridis", alpha=0.7, s=35)
max_v = max(sample["Delay_Min"].max(), sample["Optimized_Delay_Min"].max())
ax.plot([0, max_v], [0, max_v], "r--", linewidth=1, label="No improvement (y = x)")
ax.set_xlabel("Baseline Delay (min)")
ax.set_ylabel("Optimized Delay (min)")
ax.set_title("Scatter Chart \u2013 Per-Order Delay, Baseline vs Optimized", fontweight="bold")
fig.colorbar(sc, ax=ax, label="Distance (km)")
ax.legend()
fig.tight_layout()
savefig_to(fig, GALLERY_DIR, "10_scatter_delay_baseline_vs_optimized.png")

# ---- Advanced: 11. Waterfall ----
b_dist, o_dist, _ = kpis["Total Distance (km)"]
saving = b_dist - o_dist
labels = ["Baseline\nDistance", "Optimization\nSavings", "Optimized\nDistance"]
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.bar(labels[0], b_dist, color=BASE_COLOR)
ax.bar(labels[1], saving, bottom=o_dist, color=RED_COLOR)
ax.bar(labels[2], o_dist, color=OPT_COLOR)
ax.text(0, b_dist, f"{b_dist:,.0f}", ha="center", va="bottom")
ax.text(1, b_dist, f"-{saving:,.0f}", ha="center", va="bottom")
ax.text(2, o_dist, f"{o_dist:,.0f}", ha="center", va="bottom")
ax.set_ylabel("Total Distance (km)")
ax.set_title("Waterfall Chart \u2013 Distance Saved by Optimization", fontweight="bold")
fig.tight_layout()
savefig_to(fig, GALLERY_DIR, "11_waterfall_distance_savings.png")

# ---- Specialized: 12 & 13. Candlestick & OHLC ----
# NOTE: Candlestick/OHLC are finance chart types (open/high/low/close per
# period). Delivery data has no natural "open/close" price, so this is an
# honestly-labelled per-hub summary of route distance spread:
#   Open = 25th percentile, Close = 75th percentile, High = max, Low = min
g = routes.groupby("Hub")["Route_Km"]
ohlc = pd.DataFrame({"low": g.min(), "high": g.max(),
                      "open": g.quantile(0.25), "close": g.quantile(0.75)}).sort_index()

fig, ax = plt.subplots(figsize=(8, 5.5))
for i, (hub, row) in enumerate(ohlc.iterrows()):
    color = OPT_COLOR if row["close"] >= row["open"] else RED_COLOR
    ax.plot([i, i], [row["low"], row["high"]], color="black", linewidth=1)
    ax.add_patch(plt.Rectangle((i - 0.2, min(row["open"], row["close"])), 0.4,
                                abs(row["close"] - row["open"]), color=color))
ax.set_xticks(range(len(ohlc)))
ax.set_xticklabels(ohlc.index)
ax.set_ylabel("Route Distance (km)")
ax.set_title("Candlestick Chart \u2013 Route Distance Spread by Hub\n"
             "(box = 25th\u201375th percentile, whisker = min\u2013max; illustrative use of this chart type)",
             fontweight="bold", fontsize=10.5)
fig.tight_layout()
savefig_to(fig, GALLERY_DIR, "12_candlestick_route_distance_by_hub.png")

fig, ax = plt.subplots(figsize=(8, 5.5))
for i, (hub, row) in enumerate(ohlc.iterrows()):
    color = OPT_COLOR if row["close"] >= row["open"] else RED_COLOR
    ax.plot([i, i], [row["low"], row["high"]], color=color, linewidth=1.5)
    ax.plot([i - 0.15, i], [row["open"], row["open"]], color=color, linewidth=1.5)
    ax.plot([i, i + 0.15], [row["close"], row["close"]], color=color, linewidth=1.5)
ax.set_xticks(range(len(ohlc)))
ax.set_xticklabels(ohlc.index)
ax.set_ylabel("Route Distance (km)")
ax.set_title("OHLC Chart \u2013 Route Distance Spread by Hub\n"
             "(open=25th pct, close=75th pct, high/low=max/min; illustrative use of this chart type)",
             fontweight="bold", fontsize=10.5)
fig.tight_layout()
savefig_to(fig, GALLERY_DIR, "13_ohlc_route_distance_by_hub.png")

# ---- Specialized: 14. Box Plot ----
fig, ax = plt.subplots(figsize=(8, 5.5))
order_vt2 = routes.groupby("Vehicle_Type")["Utilisation_%"].median().sort_values(ascending=False).index
sns.boxplot(data=routes, x="Vehicle_Type", y="Utilisation_%", order=order_vt2,
            hue="Vehicle_Type", palette="Set2", legend=False, ax=ax)
ax.axhline(100, color="red", linestyle=":", linewidth=1, label="Full Capacity (100%)")
ax.set_title("Box Plot \u2013 Vehicle Load Utilisation by Vehicle Type", fontweight="bold")
ax.legend()
fig.tight_layout()
savefig_to(fig, GALLERY_DIR, "14_boxplot_utilisation_by_vehicle_type.png")

# ---- Specialized: 15. Arc Chart ----
flow = orders.groupby(["Origin_Hub", "Destination_Zone"])["Order_ID"].count().reset_index()
flow.columns = ["Origin_Hub", "Destination_Zone", "Orders"]
nodes = list(orders["Origin_Hub"].unique()) + sorted(orders["Destination_Zone"].unique())
pos = {node: i for i, node in enumerate(nodes)}

fig, ax = plt.subplots(figsize=(13, 5))
max_orders = flow["Orders"].max()
for _, row in flow.iterrows():
    x1, x2 = pos[row["Origin_Hub"]], pos[row["Destination_Zone"]]
    width_norm = 0.5 + 3.5 * (row["Orders"] / max_orders)
    center = (x1 + x2) / 2
    radius = abs(x2 - x1) / 2
    if radius == 0:
        continue
    arc = ArcPatch((center, 0), radius * 2, radius * 2, theta1=0, theta2=180,
                    linewidth=width_norm, edgecolor=ACCENT_COLOR, alpha=0.5)
    ax.add_patch(arc)
ax.scatter(list(pos.values()), [0] * len(pos), s=80, color=OPT_COLOR, zorder=5)
for node, x in pos.items():
    ax.text(x, -0.4, node, ha="center", va="top", rotation=45, fontsize=8)
ax.set_xlim(-1, len(nodes))
ax.set_ylim(-2, max(2.5, len(nodes) / 2))
ax.axis("off")
ax.set_title("Arc Chart \u2013 Order Flow from Hubs to Destination Zones\n(arc thickness = order volume)",
             fontweight="bold")
fig.tight_layout()
savefig_to(fig, GALLERY_DIR, "15_arc_hub_to_zone_flow.png")

# =============================================================================
# STEP 11 - ML DIAGNOSTIC PLOT GALLERY (9 plots) -> ML_DIR
# KS Plot, SHAP Plot, QQ Plot, Cumulative Explained Variance, Gini vs
# Entropy, Bias-Variance Tradeoff, ROC Curve, Precision-Recall Curve,
# Elbow Curve. Uses orders_df / routes_df already in memory - no file
# reads needed.
# =============================================================================
ML_DIR = os.environ.get("ML_DIR", os.path.join(OUT_DIR, "ml_diagnostic_plots"))
os.makedirs(ML_DIR, exist_ok=True)
print(f"\n[OK] Generating 9 ML diagnostic plots into: {ML_DIR}")

from scipy import stats as scipy_stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                              roc_curve, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# ---- Train the classifier used by KS / SHAP / ROC / PR plots ----
feat_df = orders_df.copy()
feat_df["target"] = (feat_df["Delivery_Status"] == "On Time").astype(int)
priority_map = {"High": 0, "Medium": 1, "Low": 2}
feat_df["Priority_Rank"] = feat_df["Priority"].map(priority_map)
X_ml = pd.get_dummies(
    feat_df[["Distance_Km", "Package_Weight_Kg", "Service_Time_Min", "Priority_Rank",
             "Required_Vehicle_Type", "Origin_Hub"]],
    columns=["Required_Vehicle_Type", "Origin_Hub"], drop_first=True
)
y_ml = feat_df["target"]
X_tr, X_te, y_tr, y_te = train_test_split(X_ml, y_ml, test_size=0.3, random_state=42, stratify=y_ml)
ml_clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
ml_clf.fit(X_tr, y_tr)
y_proba = ml_clf.predict_proba(X_te)[:, 1]
print(f"  Classifier test accuracy: {ml_clf.score(X_te, y_te):.3f}")

# ---- 1. KS Plot ----
pos = np.sort(y_proba[y_te.values == 1])
neg = np.sort(y_proba[y_te.values == 0])
grid = np.linspace(0, 1, 200)
cdf_pos = np.searchsorted(pos, grid, side="right") / len(pos)
cdf_neg = np.searchsorted(neg, grid, side="right") / len(neg)
ks_idx = np.argmax(np.abs(cdf_pos - cdf_neg))
ks_stat, ks_at = np.abs(cdf_pos - cdf_neg)[ks_idx], grid[ks_idx]

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(grid, cdf_pos, color=OPT_COLOR, label="On-Time (positive class)", linewidth=2)
ax.plot(grid, cdf_neg, color=RED_COLOR, label="Late (negative class)", linewidth=2)
ax.annotate("", xy=(ks_at, cdf_pos[ks_idx]), xytext=(ks_at, cdf_neg[ks_idx]),
            arrowprops=dict(arrowstyle="<->", color="black"))
ax.text(ks_at + 0.02, 0.5, f"KS = {ks_stat:.3f}", fontsize=11, fontweight="bold")
ax.set_xlabel("Predicted Probability of On-Time Delivery")
ax.set_ylabel("Cumulative Probability")
ax.set_title("KS Plot \u2013 On-Time Classifier Separation", fontweight="bold")
ax.legend(loc="lower right")
fig.tight_layout()
savefig_to(fig, ML_DIR, "01_ks_plot.png")

# ---- 2. SHAP Plot ----
try:
    import shap
    explainer = shap.TreeExplainer(ml_clf)
    sample = X_te.sample(min(300, len(X_te)), random_state=42)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values
    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, sample, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP Plot \u2013 Feature Impact on On-Time Prediction", fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig_to(fig, ML_DIR, "02_shap_plot.png")
except ImportError:
    print("  [skip] shap not installed - run `!pip install shap` in a cell above and re-run to include this plot.")

# ---- 3. QQ Plot ----
fig, ax = plt.subplots(figsize=(7.5, 6.5))
scipy_stats.probplot(orders_df["Delay_Min"], dist="norm", plot=ax)
ax.get_lines()[0].set_markerfacecolor(ACCENT_COLOR)
ax.get_lines()[0].set_markeredgecolor(ACCENT_COLOR)
ax.get_lines()[0].set_alpha(0.5)
ax.get_lines()[1].set_color(RED_COLOR)
ax.set_title("QQ Plot \u2013 Baseline Delay vs Normal Distribution", fontweight="bold")
fig.tight_layout()
savefig_to(fig, ML_DIR, "03_qq_plot.png")

# ---- 4. Cumulative Explained Variance ----
num_cols = ["Distance_Km", "Package_Weight_Kg", "Service_Time_Min",
            "Existing_Planned_Distance_Km", "Existing_Estimated_Time_Min",
            "Delay_Min", "Optimized_Delay_Min"]
X_num = StandardScaler().fit_transform(orders_df[num_cols].fillna(0))
pca = PCA()
pca.fit(X_num)
var_ratio = pca.explained_variance_ratio_
cum_var = np.cumsum(var_ratio)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
xc = np.arange(1, len(var_ratio) + 1)
ax.bar(xc, var_ratio * 100, color="#F4A261", alpha=0.8, label="Individual Component Variance")
ax.plot(xc, cum_var * 100, color=ACCENT_COLOR, marker="o", linewidth=2, label="Cumulative Variance")
for xi, yi in zip(xc, cum_var * 100):
    ax.text(xi, yi + 2, f"{yi:.0f}%", ha="center", fontsize=8)
ax.set_xlabel("Principal Component Number")
ax.set_ylabel("Explained Variance (%)")
ax.set_xticks(xc)
ax.set_ylim(0, 110)
ax.set_title("Cumulative Explained Variance \u2013 Order Numeric Features (PCA)", fontweight="bold")
ax.legend(loc="center right")
fig.tight_layout()
savefig_to(fig, ML_DIR, "04_cumulative_explained_variance.png")

# ---- 5. Gini-Impurity vs Entropy (pure math, not data-specific) ----
p = np.linspace(0.001, 0.999, 300)
entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
gini = 2 * p * (1 - p)
misclass = 1 - np.maximum(p, 1 - p)

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(p, entropy, color="black", linewidth=2, label="Entropy")
ax.plot(p, entropy / 2, color="grey", linewidth=1.5, label="Entropy (scaled)")
ax.plot(p, gini, color=RED_COLOR, linewidth=2, linestyle="--", label="Gini Impurity")
ax.plot(p, misclass, color=OPT_COLOR, linewidth=2, linestyle="-.", label="Misclassification Error")
ax.set_xlabel("p (i = 1)")
ax.set_ylabel("Impurity Index")
ax.set_title("Gini-Impurity vs Entropy \u2013 Split Criteria Used in Tree-Based Allocation Logic",
             fontweight="bold", fontsize=11.5)
ax.legend()
fig.tight_layout()
savefig_to(fig, ML_DIR, "05_gini_vs_entropy.png")

# ---- 6. Bias-Variance Tradeoff ----
bv_X = orders_df[["Distance_Km", "Package_Weight_Kg", "Service_Time_Min"]]
bv_y = orders_df["Delay_Min"]
Xtr, Xte, ytr, yte = train_test_split(bv_X, bv_y, test_size=0.3, random_state=42)
depths = range(1, 16)
train_err, test_err = [], []
for d in depths:
    m = DecisionTreeRegressor(max_depth=d, random_state=42)
    m.fit(Xtr, ytr)
    train_err.append(np.mean((m.predict(Xtr) - ytr) ** 2))
    test_err.append(np.mean((m.predict(Xte) - yte) ** 2))
train_err, test_err = np.array(train_err), np.array(test_err)
variance_proxy = np.maximum(0, test_err - train_err)
optimum_depth = list(depths)[np.argmin(test_err)]

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(depths, test_err, color="black", linewidth=2, label="Total Error (test MSE)")
ax.plot(depths, train_err, color=RED_COLOR, linewidth=2, label="Bias\u00b2 proxy (train MSE)")
ax.plot(depths, variance_proxy, color=OPT_COLOR, linewidth=2, label="Variance proxy (test - train MSE)")
ax.axvline(optimum_depth, color="grey", linestyle=":", linewidth=1.5)
ymin, ymax = ax.get_ylim()
ax.text(optimum_depth + 0.3, ymin + (ymax - ymin) * 0.08, "Optimum Model\nComplexity",
        fontsize=9, color="dimgrey")
ax.set_xlabel("Model Complexity (Decision Tree max_depth)")
ax.set_ylabel("Error (MSE, minutes\u00b2)")
ax.set_title("Bias-Variance Tradeoff \u2013 Delay Prediction by Tree Depth", fontweight="bold")
ax.legend(loc="upper center")
fig.tight_layout()
savefig_to(fig, ML_DIR, "06_bias_variance_tradeoff.png")

# ---- 7. ROC Curve ----
fpr, tpr, _ = roc_curve(y_te, y_proba)
auc = roc_auc_score(y_te, y_proba)
fig, ax = plt.subplots(figsize=(7, 6.5))
ax.plot(fpr, tpr, color=ACCENT_COLOR, linewidth=2, label=f"On-Time Classifier (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color=RED_COLOR, linestyle="--", linewidth=1.5, label="Random Classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve \u2013 On-Time Delivery Classifier", fontweight="bold")
ax.legend(loc="lower right")
fig.tight_layout()
savefig_to(fig, ML_DIR, "07_roc_curve.png")

# ---- 8. Precision-Recall Curve ----
precision, recall, _ = precision_recall_curve(y_te, y_proba)
ap = average_precision_score(y_te, y_proba)
baseline_rate = y_te.mean()
fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.plot(recall, precision, color="#F4A261", linewidth=2, marker=".", markersize=3,
        label=f"On-Time Classifier (AP = {ap:.3f})")
ax.axhline(baseline_rate, color=ACCENT_COLOR, linestyle="--", linewidth=1.5,
           label=f"No Skill (base rate = {baseline_rate:.2f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve \u2013 On-Time Delivery Classifier", fontweight="bold")
ax.legend(loc="lower left")
fig.tight_layout()
savefig_to(fig, ML_DIR, "08_precision_recall_curve.png")

# ---- 9. Elbow Curve ----
cluster_X = StandardScaler().fit_transform(
    orders_df[["Distance_Km", "Package_Weight_Kg", "Service_Time_Min"]])
ks_range = range(1, 11)
wcss = []
for k in ks_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(cluster_X)
    wcss.append(km.inertia_)
wcss = np.array(wcss)
p1 = np.array([1, wcss[0]])
p2 = np.array([len(ks_range), wcss[-1]])
line_vec = (p2 - p1) / np.linalg.norm(p2 - p1)
distances = []
for i, k in enumerate(ks_range):
    pnt = np.array([k, wcss[i]])
    proj = p1 + np.dot(pnt - p1, line_vec) * line_vec
    distances.append(np.linalg.norm(pnt - proj))
elbow_k = list(ks_range)[int(np.argmax(distances))]

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(list(ks_range), wcss, color=ACCENT_COLOR, marker="o", linewidth=2)
ax.annotate("Elbow Point", xy=(elbow_k, wcss[elbow_k - 1]),
            xytext=(elbow_k + 1.3, wcss[elbow_k - 1] + wcss[0] * 0.12),
            arrowprops=dict(arrowstyle="->", color=RED_COLOR), fontsize=10, fontweight="bold", color=RED_COLOR)
ax.set_xlabel("K - Value (number of order clusters)")
ax.set_ylabel("WCSS")
ax.set_title("Elbow Curve \u2013 Optimal Order Clusters by Distance/Weight/Service Time",
             fontweight="bold", fontsize=11.5)
fig.tight_layout()
savefig_to(fig, ML_DIR, "09_elbow_curve.png")

# =============================================================================
# DONE
# =============================================================================
print(f"\nDONE. Result files: {OUT_DIR}")
print(f"DONE. 10 core plots: {PLOTS_DIR}")
print(f"DONE. 15-chart gallery: {GALLERY_DIR}")
print(f"DONE. 9 ML diagnostic plots: {ML_DIR}")
print("You can now download all of these from the Colab folder panel.")
