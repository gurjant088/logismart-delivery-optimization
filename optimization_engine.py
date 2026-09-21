# -*- coding: utf-8 -*-
"""
LogiSmart India - Smart Delivery Route and Resource Optimization System
Core allocation + routing optimization engine (prototype logic).

This module implements the algorithmic core described in the project workbook:
  - Data loading & validation           (Activity 2)
  - Data structures (dict/graph/queue)  (Activity 3)
  - Allocation + routing algorithms     (Activity 4)
  - Optimization model                  (Activity 5)
  - Baseline vs optimized KPIs          (Activity 8, 9)

Design mirrors the Java class layout suggested in the workbook:
  Order, Vehicle, DataLoader, Validator, AllocationService, RouteOptimizer, ResultService
Implemented here in Python for rapid prototyping/validation; the same class
boundaries and logic translate directly to Java (see README notes at bottom).
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict

pd.set_option('display.width', 120)

# ---------------------------------------------------------------------------
# 1. DATA LOADER  (Activity 2 / Activity 6 -> DataLoader class)
# ---------------------------------------------------------------------------
import os
import glob

FILENAME = "Competitor_Beta_Dataset_for_Students.xlsx"

def find_data_file(filename):
    """Search common Colab/local locations so this runs regardless of where
    the file was uploaded, instead of failing on a single hardcoded path."""
    candidates = [
        filename,                                  # current working directory
        os.path.join("/content", filename),
        os.path.join("/content/sample_data", filename),
        os.path.join("/mnt/user-data/uploads", filename),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # last resort: recursive search under /content (Colab) or cwd
    search_roots = ["/content", "."]
    for root in search_roots:
        if os.path.isdir(root):
            matches = glob.glob(os.path.join(root, "**", filename), recursive=True)
            if matches:
                return matches[0]
    raise FileNotFoundError(
        f"Could not find '{filename}'. In Colab, click the folder icon on the "
        f"left sidebar, upload the file there, then re-run this cell. "
        f"Checked: {candidates}"
    )

DATA_PATH = find_data_file(FILENAME)
print(f"Loading data from: {DATA_PATH}")

orders_df = pd.read_excel(DATA_PATH, sheet_name="Delivery Orders")
vehicles_df = pd.read_excel(DATA_PATH, sheet_name="Vehicle Master")
dist_df = pd.read_excel(DATA_PATH, sheet_name="Zone Distance Matrix", index_col=0)

# ---------------------------------------------------------------------------
# 2. VALIDATOR  (Activity 2 -> Validator class)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 3. DATA STRUCTURES  (Activity 3)
# ---------------------------------------------------------------------------
# Vehicle type hierarchy: a vehicle can carry any order whose required type
# rank is <= its own rank (a Van can carry a Bike-class order, etc.)
TYPE_RANK = {"Bike": 0, "Van": 1, "Mini Truck": 2}

# Vehicle "fleet" as an in-memory object list, grouped by hub (dict of lists) -
# mirrors a Java Map<String, List<Vehicle>>
fleet_by_hub = defaultdict(list)
for _, v in vehicles_df.iterrows():
    fleet_by_hub[v['Hub']].append({
        "Vehicle_ID": v['Vehicle_ID'],
        "Hub": v['Hub'],
        "Vehicle_Type": v['Vehicle_Type'],
        "Type_Rank": TYPE_RANK[v['Vehicle_Type']],
        "Capacity_Kg": v['Capacity_Kg'],
        "Cost_per_Km_INR": v['Cost_per_Km_INR'],
        "Max_Working_Minutes": v['Max_Working_Minutes'],
        "Available": v['Availability'] == "Available",
        "Remaining_Capacity": v['Capacity_Kg'] if v['Availability'] == "Available" else 0,
        "Remaining_Minutes": v['Max_Working_Minutes'] if v['Availability'] == "Available" else 0,
        "Assigned_Orders": [],   # route stops (destination zones) - list acts as a queue/stack
    })

# Zone/hub distance graph -> adjacency matrix (dict of dict), used for nearest-neighbor routing
graph = {i: {j: dist_df.loc[i, j] for j in dist_df.columns} for i in dist_df.index}

# Average travel speed derived from the existing plan, used to translate km -> minutes
# for the optimized route (avoids inventing an arbitrary constant)
orders_df['_travel_min_existing'] = orders_df['Existing_Estimated_Time_Min'] - orders_df['Service_Time_Min']
AVG_SPEED_KMPH = float(
    (orders_df['Existing_Planned_Distance_Km'].sum() / (orders_df['_travel_min_existing'].sum() / 60.0))
)

# ---------------------------------------------------------------------------
# 4. ALLOCATION SERVICE  (Activity 4/5 -> AllocationService)
#
# ASSUMPTION (documented, since the dataset carries no date/day field):
# 1,800 orders far exceed the fleet's single-shift capacity (52 vehicles x
# 480 working minutes), so the order volume is treated as a multi-trip
# dispatch backlog: each vehicle can complete several sequential trips
# (returning to hub and reloading) across the review period, rather than
# exactly one trip each. This is the standard real-world interpretation of
# "improve resource utilisation" for a fleet this size. Bin packing is used
# to build trips (First-Fit-Decreasing by weight, priority-ordered), and
# trips are then round-robined across the hub's available vehicles of the
# matching type so utilisation and vehicle counts stay realistic.
# ---------------------------------------------------------------------------
PRIORITY_RANK = {"High": 0, "Medium": 1, "Low": 2}

def build_trips():
    """Bin-pack orders into trips per (hub, vehicle type), respecting weight
    capacity and an estimated working-minutes budget, priority-ordered."""
    trips = []          # list of dicts: hub, type, orders[], weight, minutes
    unallocated = []

    orders_sorted = orders_df.copy()
    orders_sorted['_prio_rank'] = orders_sorted['Priority'].map(PRIORITY_RANK)
    orders_sorted = orders_sorted.sort_values(
        by=['_prio_rank', 'Package_Weight_Kg'], ascending=[True, False]
    )

    # group by (hub, matching vehicle type) - use the smallest capable type
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
                unallocated.append({"Order_ID": order['Order_ID'],
                                     "Reason": f"No available {vtype} at {hub}"})
            continue
        capacity = vehicles_of_type[0]['Capacity_Kg']
        max_minutes = vehicles_of_type[0]['Max_Working_Minutes']

        bins = []  # each: {weight, minutes, orders:[]}
        for order in group_orders:  # already priority/weight sorted (FFD)
            weight = order['Package_Weight_Kg']
            service_time = order['Service_Time_Min']
            # conservative travel-time proxy (hub -> stop, before consolidation savings)
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

        # round-robin trips across the hub's available vehicles of this type
        for i, b in enumerate(bins):
            vehicle = vehicles_of_type[i % len(vehicles_of_type)]
            trips.append({
                "Vehicle_ID": vehicle['Vehicle_ID'], "Hub": hub, "Vehicle_Type": vtype,
                "Capacity_Kg": capacity, "Cost_per_Km_INR": vehicle['Cost_per_Km_INR'],
                "Trip_No": i + 1, "Orders": b['orders'],
            })

    return trips, unallocated

trips, unallocated_orders = build_trips()

# populate fleet_by_hub-style structure for downstream routing (kept for API shape)
for t in trips:
    for order in t['Orders']:
        pass  # routing reads directly from `trips` below

# ---------------------------------------------------------------------------
# 5. ROUTE OPTIMIZER  (Activity 4/5 -> RouteOptimizer)
#    Nearest-neighbour multi-stop routing per vehicle trip
# ---------------------------------------------------------------------------
def nearest_neighbour_route(hub, stops):
    """stops: list of destination zone names (may repeat). Returns total km."""
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
for idx, t in enumerate(trips):
    stops = [o['Destination_Zone'] for o in t['Orders']]
    route_km = nearest_neighbour_route(t['Hub'], stops)
    load = sum(o['Package_Weight_Kg'] for o in t['Orders'])
    route_records.append({
        "Trip_ID": f"{t['Vehicle_ID']}-T{t['Trip_No']}",
        "Vehicle_ID": t['Vehicle_ID'],
        "Hub": t['Hub'],
        "Vehicle_Type": t['Vehicle_Type'],
        "Stops": len(stops),
        "Route_Km": round(route_km, 2),
        "Load_Kg": round(load, 2),
        "Capacity_Kg": t['Capacity_Kg'],
        "Utilisation_%": round(100 * load / t['Capacity_Kg'], 1),
        "Estimated_Cost_INR": round(route_km * t['Cost_per_Km_INR'], 2),
    })

routes_df = pd.DataFrame(route_records)

# ---------------------------------------------------------------------------
# 6. RESULT SERVICE / KPI COMPARISON  (Activity 8, 9)
# ---------------------------------------------------------------------------
# --- Baseline (current process: one independent trip per order, no consolidation) ---
baseline_total_distance = orders_df['Existing_Planned_Distance_Km'].sum()
baseline_avg_delay = orders_df['Delay_Min'].mean()
baseline_on_time_rate = (orders_df['Delivery_Status'] == 'On Time').mean()
baseline_total_cost = float((
    orders_df.assign(rank=orders_df['Required_Vehicle_Type'].map(TYPE_RANK))
    .merge(vehicles_df.assign(rank=vehicles_df['Vehicle_Type'].map(TYPE_RANK))
           .groupby('rank')['Cost_per_Km_INR'].first().reset_index(), on='rank')
)['Existing_Planned_Distance_Km'] .mul((
    orders_df.assign(rank=orders_df['Required_Vehicle_Type'].map(TYPE_RANK))
    .merge(vehicles_df.assign(rank=vehicles_df['Vehicle_Type'].map(TYPE_RANK))
           .groupby('rank')['Cost_per_Km_INR'].first().reset_index(), on='rank')
)['Cost_per_Km_INR']).sum())

# --- Optimized (consolidated multi-drop routes from allocation + routing engine) ---
optimized_total_distance = routes_df['Route_Km'].sum()
optimized_total_cost = routes_df['Estimated_Cost_INR'].sum()
allocation_rate = 1 - (len(unallocated_orders) / len(orders_df))
avg_vehicle_utilisation = routes_df['Utilisation_%'].mean()

# Optimized delay/time: travel time scales with the distance actually driven;
# time saved per order is applied against its existing delay, floored at 0.
order_to_trip = {}
for idx, t in enumerate(trips):
    trip_id = f"{t['Vehicle_ID']}-T{t['Trip_No']}"
    for o in t['Orders']:
        order_to_trip[o['Order_ID']] = trip_id

trip_route_km = dict(zip(routes_df['Trip_ID'], routes_df['Route_Km']))
trip_stop_count = dict(zip(routes_df['Trip_ID'], routes_df['Stops']))

def optimized_delay_for_order(row):
    trip_id = order_to_trip.get(row['Order_ID'])
    if trip_id is None:
        return row['Delay_Min']  # unallocated -> cannot improve, kept as-is
    # this order's share of its trip's consolidated route distance
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

# ---------------------------------------------------------------------------
# 7. OUTPUT (for reporting / xlsx population)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== VALIDATION ===")
    for name, val in validation_results:
        print(f"{name}: {val}")

    print("\n=== BASELINE vs OPTIMIZED KPIs ===")
    for k, (b, o, imp) in kpi_summary.items():
        print(f"{k}: baseline={b}, optimized={o}, improvement={imp}")

    print("\nUnallocated orders:", len(unallocated_orders))
    print(routes_df.describe(include='all').T[['count', 'mean']] if not routes_df.empty else "no routes")

    # persist artifacts for xlsx population step - write next to the input file
    # (works both locally and in Colab) instead of a hardcoded /home/claude path
    OUT_DIR = os.path.dirname(DATA_PATH) or "."
    routes_df.to_csv(os.path.join(OUT_DIR, "routes_output.csv"), index=False)
    orders_df.to_csv(os.path.join(OUT_DIR, "orders_with_optimized_delay.csv"), index=False)
    with open(os.path.join(OUT_DIR, "kpi_summary.json"), "w") as f:
        json.dump({k: list(v) for k, v in kpi_summary.items()}, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "validation_results.json"), "w") as f:
        json.dump(validation_results, f, indent=2)
    with open(os.path.join(OUT_DIR, "unallocated.json"), "w") as f:
        json.dump(unallocated_orders, f, indent=2, default=str)
    print(f"\nArtifacts written to: {OUT_DIR}")
