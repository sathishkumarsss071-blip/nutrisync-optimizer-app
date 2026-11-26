import streamlit as st
import pandas as pd
import pulp

# ---------------------------------------------------------
# 1. Page Configuration & Header
# ---------------------------------------------------------
st.set_page_config(
    page_title="NutriSync — Automated Macro & Grocery Optimizer",
    page_icon="🥗",
    layout="wide"
)

st.title("🥗 NutriSync — Automated Macro & Grocery Optimizer")
st.caption("SIMATS Engineering Capstone Project | Python Programming (CSA0829) | Guided by Shri Rajeshwaran M")
st.markdown("---")

# ---------------------------------------------------------
# 2. Staple Ingredient Database (Per 100g / Standard Serving)
# ---------------------------------------------------------
STAPLE_DATA = [
    {"Ingredient": "Rolled Oats", "Aisle": "Grains & Cereals", "Dietary": "Veg", "Calories": 389, "Protein_g": 16.9, "Cost_INR": 18, "Max_Servings": 2.5},
    {"Ingredient": "Chicken Breast", "Aisle": "Lean Proteins", "Dietary": "Non-Veg", "Calories": 165, "Protein_g": 31.0, "Cost_INR": 32, "Max_Servings": 3.5},
    {"Ingredient": "Soya Chunks", "Aisle": "Lean Proteins", "Dietary": "Veg", "Calories": 345, "Protein_g": 52.0, "Cost_INR": 16, "Max_Servings": 1.5},
    {"Ingredient": "Whole Eggs (2 units)", "Aisle": "Dairy & Eggs", "Dietary": "Non-Veg", "Calories": 143, "Protein_g": 12.6, "Cost_INR": 14, "Max_Servings": 2.5},
    {"Ingredient": "Paneer (Low Fat)", "Aisle": "Dairy & Eggs", "Dietary": "Veg", "Calories": 265, "Protein_g": 18.3, "Cost_INR": 38, "Max_Servings": 2.0},
    {"Ingredient": "Brown Rice & Toor Dal", "Aisle": "Grains & Pulses", "Dietary": "Veg", "Calories": 350, "Protein_g": 14.2, "Cost_INR": 12, "Max_Servings": 3.0},
    {"Ingredient": "Greek Yogurt / Curd", "Aisle": "Dairy & Eggs", "Dietary": "Veg", "Calories": 97, "Protein_g": 9.0, "Cost_INR": 20, "Max_Servings": 2.5},
    {"Ingredient": "Spinach & Broccoli", "Aisle": "Fresh Produce", "Dietary": "Veg", "Calories": 34, "Protein_g": 2.8, "Cost_INR": 10, "Max_Servings": 3.0},
    {"Ingredient": "Roasted Peanuts", "Aisle": "Nuts & Seeds", "Dietary": "Veg", "Calories": 567, "Protein_g": 25.8, "Cost_INR": 22, "Max_Servings": 0.8},
    {"Ingredient": "Banana", "Aisle": "Fresh Produce", "Dietary": "Veg", "Calories": 89, "Protein_g": 1.1, "Cost_INR": 8, "Max_Servings": 2.0},
]

df_all = pd.DataFrame(STAPLE_DATA)

# ---------------------------------------------------------
# 3. Sidebar: User Targets & Pantry Constraints
# ---------------------------------------------------------
with st.sidebar:
    st.header("🎯 Daily Nutrition Targets")
    target_cal = st.slider("Daily Calorie Target (kcal)", min_value=1400, max_value=3500, value=2200, step=50)
    target_prot = st.slider("Minimum Daily Protein (g)", min_value=60, max_value=220, value=140, step=5)
    days = st.selectbox("Planning Horizon (Days)", options=[7, 14, 30], index=0)

    st.header("🥦 Dietary & Pantry Filter")
    diet_mode = st.radio("Dietary Preference", ["All Staples (Non-Veg + Veg)", "Vegetarian Only"])
    
    if diet_mode == "Vegetarian Only":
        df_filtered = df_all[df_all["Dietary"] == "Veg"].reset_index(drop=True)
    else:
        df_filtered = df_all.copy().reset_index(drop=True)

    selected_items = st.multiselect(
        "Available Pantry Staples",
        options=df_filtered["Ingredient"].tolist(),
        default=df_filtered["Ingredient"].tolist()
    )

df = df_filtered[df_filtered["Ingredient"].isin(selected_items)].reset_index(drop=True)

# ---------------------------------------------------------
# 4. Linear Programming Optimization Engine (PuLP)
# ---------------------------------------------------------
def solve_meal_plan(pantry_df: pd.DataFrame, cal_goal: float, prot_goal: float, horizon_days: int):
    if pantry_df.empty:
        return "Empty", None, {}

    model = pulp.LpProblem("NutriSync_Optimizer", pulp.LpMinimize)

    # Decision Variables: Daily servings (in 100g units)
    qty = {
        i: pulp.LpVariable(f"qty_{i}", lowBound=0, upBound=pantry_df.loc[i, "Max_Servings"])
        for i in pantry_df.index
    }

    # Objective Function: Minimize daily ingredient cost
    model += pulp.lpSum(pantry_df.loc[i, "Cost_INR"] * qty[i] for i in pantry_df.index), "Minimize_Daily_Cost"

    # Constraint 1: Daily Calories within ±3% of target
    model += pulp.lpSum(pantry_df.loc[i, "Calories"] * qty[i] for i in pantry_df.index) >= cal_goal * 0.97, "Min_Calories"
    model += pulp.lpSum(pantry_df.loc[i, "Calories"] * qty[i] for i in pantry_df.index) <= cal_goal * 1.03, "Max_Calories"

    # Constraint 2: Daily Protein >= Target
    model += pulp.lpSum(pantry_df.loc[i, "Protein_g"] * qty[i] for i in pantry_df.index) >= prot_goal, "Min_Protein"

    # Constraint 3: Minimum Fresh Produce for Micronutrients (at least 100g/day)
    produce_idx = pantry_df[pantry_df["Aisle"] == "Fresh Produce"].index
    if len(produce_idx) > 0:
        model += pulp.lpSum(qty[i] for i in produce_idx) >= 1.0, "Min_Fresh_Produce"

    # Solve using CBC
    status_code = model.solve(pulp.PULP_CBC_CMD(msg=False))
    status = pulp.LpStatus[status_code]

    if status != "Optimal":
        return status, None, {}

    # Extract Results
    results = []
    exact_cal = 0.0
    exact_prot = 0.0
    exact_daily_cost = 0.0

    for i in pantry_df.index:
        daily_units = qty[i].varValue
        if daily_units and daily_units > 0.05:  # Filter negligible traces (< 5g)
            exact_cal += daily_units * pantry_df.loc[i, "Calories"]
            exact_prot += daily_units * pantry_df.loc[i, "Protein_g"]
            exact_daily_cost += daily_units * pantry_df.loc[i, "Cost_INR"]

            daily_grams = round(daily_units * 100, 1)
            weekly_grams = round(daily_grams * horizon_days, 1)
            results.append({
                "Aisle": pantry_df.loc[i, "Aisle"],
                "Ingredient": pantry_df.loc[i, "Ingredient"],
                "Daily_Qty_g": daily_grams,
                f"{horizon_days}_Day_Qty_g": weekly_grams,
                f"{horizon_days}_Day_Qty_kg": round(weekly_grams / 1000, 2),
                "Daily_Calories": round(daily_units * pantry_df.loc[i, "Calories"], 1),
                "Daily_Protein_g": round(daily_units * pantry_df.loc[i, "Protein_g"], 1),
                f"{horizon_days}_Day_Cost_INR": round(daily_units * pantry_df.loc[i, "Cost_INR"] * horizon_days, 2),
            })

    if not results:
        return "Infeasible", None, {}

    res_df = pd.DataFrame(results).sort_values(by=["Aisle", "Ingredient"]).reset_index(drop=True)
    summary = {
        "achieved_cal": round(exact_cal, 1),
        "achieved_prot": round(exact_prot, 1),
        "total_cost": round(exact_daily_cost * horizon_days, 2),
        "total_weight_kg": round(res_df[f"{horizon_days}_Day_Qty_kg"].sum(), 2),
    }
    return status, res_df, summary

# ---------------------------------------------------------
# 5. Dashboard Output & Interactive Visualization
# ---------------------------------------------------------
status, res_df, summary = solve_meal_plan(df, target_cal, target_prot, days)

if status == "Optimal" and res_df is not None and not res_df.empty:
    # Top KPI Metrics Row
    cal_diff = summary["achieved_cal"] - target_cal
    prot_diff = summary["achieved_prot"] - target_prot

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Achieved Daily Calories", f"{summary['achieved_cal']} kcal", f"{cal_diff:+.1f} kcal")
    col2.metric("Achieved Daily Protein", f"{summary['achieved_prot']} g", f"{prot_diff:+.1f} g")
    col3.metric(f"{days}-Day Grocery Cost", f"₹ {summary['total_cost']}")
    col4.metric(f"Total {days}-Day Weight", f"{summary['total_weight_kg']} kg")

    st.markdown(f"### 🛒 Auto-Sorted {days}-Day Grocery List (Grouped by Aisle)")
    st.dataframe(res_df, use_container_width=True, hide_index=True)

    # Chart & CSV Export Columns
    left_col, right_col = st.columns([1.4, 1])

    with left_col:
        st.markdown("### 📊 Daily Protein Contribution by Supermarket Aisle")
        aisle_summary = res_df.groupby("Aisle")["Daily_Protein_g"].sum().sort_values(ascending=True)
        st.bar_chart(aisle_summary, color="#0f766e")

    with right_col:
        st.markdown("### 📥 Export Grocery Checklist")
        st.info("Download your aisle-sorted grocery list as a CSV file ready for mobile shopping.")
        csv_data = res_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download {days}-Day Grocery List (.CSV)",
            data=csv_data,
            file_name=f"nutrisync_{days}day_grocery_list.csv",
            mime="text/csv",
            type="primary"
        )
        st.markdown("**Team Members:** Retesh.B (1924260381), Sathish kumar.S (1921260200), Binon.M (1912260728)")

elif status == "Empty":
    st.warning("⚠️ Please select at least one staple ingredient in the sidebar.")
else:
    st.error(f"❌ Optimization Infeasible ({status}): Cannot satisfy {target_cal} kcal and {target_prot}g protein with the selected staples. Try increasing calorie bounds or adding high-protein staples.")
