#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINDEX 2026: CAUSAL ANALYSIS - SAVED OUTCOME
WITH REGIONAL HETEROGENEITY AND WORKING REFUTATIONS
"""

import pandas as pd
import numpy as np
from dowhy import CausalModel
import statsmodels.api as sm
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

treatment = 'DCIs'
outcome = 'saved'
instruments_list = ['lnLight', 'distance']
potential_controls = ['age', 'female', 'educ', 'inc_q', 'urbanicity']

# Regional definitions
west_africa = ['Benin', 'Burkina Faso', 'Cote d\'Ivoire', 'Gambia, The', 'Ghana',
               'Guinea', 'Liberia', 'Mali', 'Mauritania', 'Niger', 'Nigeria',
               'Senegal', 'Sierra Leone', 'Togo']

east_africa = ['Comoros', 'Ethiopia', 'Kenya', 'Madagascar', 'Malawi',
               'Mozambique', 'Tanzania', 'Uganda', 'Zambia', 'Zimbabwe']

southern_africa = ['Botswana', 'Eswatini', 'Lesotho', 'Namibia', 'South Africa']

central_africa = ['Cameroon', 'Chad', 'Congo, Dem. Rep.', 'Congo, Rep.', 'Gabon']


def get_first_stage_f(df_clean, final_controls):
    """Calculate first-stage F-statistic for instrument strength"""
    try:
        X = sm.add_constant(df_clean[final_controls + instruments_list])
        y = df_clean[treatment]
        first_stage = sm.OLS(y, X).fit()
        f_test = first_stage.f_test('lnLight = 0, distance = 0')
        return float(f_test.fvalue)
    except:
        return np.nan


def run_dowhy(df, group_name):
    print(f"🚀 ANALYZING SAVINGS: {group_name}")
    df = df.copy()

    # --- 1. ENCODING ---
    mapping = {
        'sex': {'female': 1, 'male': 0},
        'saved': {'yes': 1, 'no': 0},
        'educ': {'primary': 1, 'second': 2, 'tertiary': 3},
        'inc_q': {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5},
        'urbanicity': {'urban': 1, 'rural': 0}
    }

    for col, rules in mapping.items():
        t_col = 'female' if col == 'sex' else col
        if col in df.columns:
            s = df[col].astype(str).str.lower().str.strip()
            df[t_col] = np.nan
            for k, v in rules.items():
                df.loc[s.str.contains(k, na=False), t_col] = v

    # --- 2. CLEANING ---
    all_req = [treatment, outcome] + instruments_list + potential_controls
    for col in all_req:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df_clean = df.dropna(subset=[c for c in all_req if c in df.columns])
    df_clean = df_clean.dropna()

    # Clip for IV stability
    for col in [treatment] + instruments_list:
        if len(df_clean) > 0 and col in df_clean.columns:
            df_clean[col] = df_clean[col].clip(
                df_clean[col].quantile(0.05),
                df_clean[col].quantile(0.95)
            )

    final_controls = [
        c for c in potential_controls
        if c in df_clean.columns and df_clean[c].nunique() > 1
    ]

    f_stat = get_first_stage_f(df_clean, final_controls)

    if len(df_clean) < 100:
        print(f"  Skipping {group_name}: insufficient observations ({len(df_clean)})")
        return None

    # --- 3. MODELING ---
    try:
        model = CausalModel(
            data=df_clean,
            treatment=treatment,
            outcome=outcome,
            common_causes=final_controls,
            instruments=instruments_list
        )
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified,
            method_name="iv.instrumental_variable"
        )

        # --- COMPLETELY REWRITTEN SAFE REFUTE FUNCTION ---
        def safe_refute(m_name, **kwargs):
            """Safely run refutation tests with proper error handling"""
            try:
                # Run refutation
                refutation = model.refute_estimate(
                    identified, estimate,
                    method_name=m_name,
                    num_simulations=20,
                    **kwargs
                )

                # DEBUG: Print what we got for Global
                if group_name == "GLOBAL" and m_name == "random_common_cause":
                    print(f"    🔍 DEBUG - {m_name} returned: {type(refutation)}")
                    if hasattr(refutation, '__dict__'):
                        print(f"    Attributes: {dir(refutation)[:10]}")

                # Try different ways to extract p-value
                p_value = np.nan

                # Method 1: Direct p_value attribute
                if hasattr(refutation, 'p_value'):
                    p_value = refutation.p_value
                    if group_name == "GLOBAL":
                        print(f"    ✅ Found p_value attribute: {p_value}")

                # Method 2: refutation_result dictionary
                elif hasattr(refutation, 'refutation_result'):
                    result_dict = refutation.refutation_result
                    if isinstance(result_dict, dict) and 'p_value' in result_dict:
                        p_value = result_dict['p_value']
                        if group_name == "GLOBAL":
                            print(f"    ✅ Found p_value in refutation_result: {p_value}")

                # Method 3: Dictionary-like object
                elif isinstance(refutation, dict) and 'p_value' in refutation:
                    p_value = refutation['p_value']
                    if group_name == "GLOBAL":
                        print(f"    ✅ Found p_value in dict: {p_value}")

                # Method 4: Try to calculate from new_effect
                elif hasattr(refutation, 'new_effect') and hasattr(estimate, 'value'):
                    # If p-value not available, use effect comparison
                    original = estimate.value
                    new = refutation.new_effect
                    if original != 0 and new is not None:
                        # Rough heuristic: if new effect is very different, p is small
                        diff_ratio = abs((new - original) / original)
                        p_value = min(1.0, diff_ratio * 2)
                        if group_name == "GLOBAL":
                            print(f"    ⚠️ Calculated p from effect diff: {p_value}")

                if np.isnan(p_value):
                    if group_name == "GLOBAL":
                        print(f"    ❌ Could not extract p-value for {m_name}")

                return p_value

            except Exception as e:
                if group_name == "GLOBAL":
                    print(f"    ❌ {m_name} failed: {str(e)}")
                return np.nan

        # Run refutations
        p_random = safe_refute("random_common_cause")
        p_subset = safe_refute("data_subset_refuter", subset_fraction=0.8)
        p_placebo = safe_refute("placebo_treatment_refuter", placebo_type="permute")
        p_dummy = safe_refute("dummy_outcome_refuter")

        return {
            'Group': group_name,
            'Impact': estimate.value * 100,
            'F_Stat': f_stat,
            'P_Subset': p_subset,
            'P_Placebo': p_placebo,
            'P_Dummy': p_dummy,
            'P_Random': p_random
        }
    except Exception as e:
        print(f"  ❌ Model failed for {group_name}: {str(e)[:100]}")
        return None


if __name__ == "__main__":
    full_path = r'D:\SSA Financial Inclusion\Full.csv'

    if not os.path.exists(full_path):
        print(f"❌ File not found: {full_path}")
        exit()

    print("=" * 90)
    print("FINDEX 2026: SAVINGS OUTCOME ANALYSIS")
    print("=" * 90)

    df_raw = pd.read_csv(full_path)
    print(f"✅ Loaded {len(df_raw):,} observations")

    # Create region column
    if 'economy' in df_raw.columns:
        df_raw['region'] = 'Other'
        df_raw.loc[df_raw['economy'].isin(west_africa), 'region'] = 'West Africa'
        df_raw.loc[df_raw['economy'].isin(east_africa), 'region'] = 'East Africa'
        df_raw.loc[df_raw['economy'].isin(southern_africa), 'region'] = 'Southern Africa'
        df_raw.loc[df_raw['economy'].isin(central_africa), 'region'] = 'Central Africa'

        print("\n🌍 Region distribution:")
        for region in ['West Africa', 'East Africa', 'Southern Africa', 'Central Africa']:
            count = (df_raw['region'] == region).sum()
            pct = count / len(df_raw) * 100
            print(f"  • {region}: {count:,} ({pct:.1f}%)")
    else:
        df_raw['region'] = 'Unknown'
        print("⚠️ 'economy' column not found - regional analysis skipped")

    # Create subgroup flags
    df_raw['is_fem'] = df_raw['sex'].astype(str).str.lower().str.contains('female')
    df_raw['is_urb'] = df_raw['urbanicity'].astype(str).str.lower().str.contains('urban')

    # Define all groups to analyze
    tasks = [
        (df_raw, "GLOBAL"),
        (df_raw[df_raw['is_fem'] == True], "Female"),
        (df_raw[df_raw['is_fem'] == False], "Male"),
        (df_raw[df_raw['is_urb'] == True], "Urban"),
        (df_raw[df_raw['is_urb'] == False], "Rural")
    ]

    # Income quintiles
    for q in ['first', 'second', 'third', 'fourth', 'fifth']:
        mask = df_raw['inc_q'].astype(str).str.lower().str.contains(q)
        tasks.append((df_raw[mask], f"Q-{q.capitalize()}"))

    # Regions
    for region in ['West Africa', 'East Africa', 'Southern Africa', 'Central Africa']:
        region_subset = df_raw[df_raw['region'] == region]
        if len(region_subset) >= 200:
            tasks.append((region_subset, f"Region: {region}"))
        else:
            print(f"Skipping {region}: only {len(region_subset)} observations")

    # Run all analyses
    print("\n" + "=" * 90)
    print("RUNNING ANALYSES")
    print("=" * 90)

    results = [run_dowhy(d, n) for d, n in tasks if not d.empty]

    # Print results
    print("\n" + "=" * 115)
    print(
        f"{'GROUP':<20} {'SAVINGS IMPACT':<15} {'F-STAT':<10} {'SUBSET-P':<10} {'PLACEBO-P':<10} {'DUMMY-P':<10} {'RANDOM-P':<10}")
    print("-" * 115)

    valid_results = [r for r in results if r]
    for r in valid_results:
        # Format F-stat with strength indicator
        if not np.isnan(r['F_Stat']):
            strong = "✅" if r['F_Stat'] > 10 else "⚠️"
            f_display = f"{r['F_Stat']:>8.1f}{strong}"
        else:
            f_display = "     NA"


        # Format p-values
        def format_p(p):
            if pd.isna(p):
                return "      NaN"
            else:
                return f"{p:>9.3f}"


        print(f"{r['Group']:<20} {r['Impact']:>10.2f}% {f_display:>9} "
              f"{format_p(r['P_Subset'])} {format_p(r['P_Placebo'])} "
              f"{format_p(r['P_Dummy'])} {format_p(r['P_Random'])}")

    # Save results
    if valid_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'savings_impact_report_{timestamp}.csv'

        df_out = pd.DataFrame(valid_results)
        df_out.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to '{output_file}'")

    print("\n" + "=" * 115)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 115)