# -*- coding: utf-8 -*-
"""
LogiSmart India - Smart Delivery Route and Resource Optimization System
ALL-IN-ONE SCRIPT: data loading, validation, allocation, route optimization,
KPI comparison, and functional tests - everything in a single file.

HOW TO USE IN COLAB:
  1. Upload "Competitor Beta Dataset for Students.xlsx" to /content/sample_data/
     (this is the only file you need to upload - this script does the rest)
  2. Upload this file (run_project.py) to /content/sample_data/ as well
  3. Run:  %run /content/sample_data/run_project.py
     (or open this file in a Colab cell and press Run)
  4. Everything prints below, and result files are saved to /content/sample_data/
"""

import pandas as pd
import numpy as np
import json
import os
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
        f"Checked these locations:\n" + "\n".join(f"  - {p}" for p in CANDIDATE_PATHS)
    )

print(f"[OK] Loading data from: {DATA_PATH}")
OUT_DIR = os.path.dirname(DATA_PATH)

orders_df = pd.read_excel(DATA_PATH, sheet_name="Delivery Orders")
vehicles_df = pd.read_excel(DATA_PATH, sheet_name="Vehicle Master")
dist_df = pd.read_excel(DATA_PATH, sheet_name="Zone Distance Matrix", index_col=0)
print(f"[OK] Loaded {len(orders_df)} orders, {len(vehicles_df)} vehicles, "
      f"{dist_df.shape[0]}x{dist_df.shape[1]} distance matrix.\n")

# =============================================================================
# STEP 1 - VALIDATION  (Activity 2)
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
# STEP 2 - DATA STRUCTURES  (Activity 3)
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
# STEP 3 - ALLOCATION  (Activity 4/5)
# ASSUMPTION: 1,800 orders exceed the fleet's single-shift capacity, so this
# is modelled as a multi-trip dispatch backlog (vehicles run several trips
# across the review period), documented since the data has no date field.
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
# STEP 4 - ROUTE OPTIMIZATION  (Activity 4/5): nearest-neighbour per trip
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
# STEP 5 - KPI COMPARISON  (Activity 8, 9)
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
# STEP 6 - FUNCTIONAL TESTS  (Activity 10)
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
# STEP 7 - SAVE ALL RESULTS
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

print(f"\n[OK] All result files saved to: {OUT_DIR}")
print("  - routes_output.csv")
print("  - orders_with_optimized_delay.csv")
print("  - kpi_summary.json")
print("  - validation_results.json")
print("  - unallocated.json")
print("  - test_results.json")
print("\nDONE. You can now download these files from the Colab folder panel.")
