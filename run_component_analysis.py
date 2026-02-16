#!/usr/bin/env python
"""
Quick script to run component analysis

This isolates the impact of each HMB treatment component (NSAID, TXA, pill, hIUD)
on anemia reduction by comparing scenarios where each treatment is offered alone.
"""

import sciris as sc
from component_analysis import (
    run_component_analysis,
    calculate_component_impacts,
    print_component_summary,
    plot_component_impacts,
    load_component_results
)


def main():
    """Run complete component analysis."""

    # Check if results already exist
    try:
        print('Checking for existing results...')
        results = load_component_results()
        print('Found existing results! Loading from disk...\n')
    except:
        print('No existing results found. Running new simulations...\n')
        # Run simulations (this will take some time)
        results = run_component_analysis(n_runs=10, save_results=True)

    # Calculate impacts
    print('\nCalculating component impacts...')
    impacts = calculate_component_impacts(results)

    # Print summary
    print_component_summary(impacts)

    # Create visualization
    print('Creating visualization...')
    plot_component_impacts(impacts)

    # Save impacts as CSV for easy reference
    print('\nSaving impact summary...')
    sc.path('results').mkdir(exist_ok=True)

    # Create summary dataframe
    import pandas as pd
    summary_data = []
    for component in ['baseline', 'nsaid', 'txa', 'pill', 'hiud', 'full']:
        summary_data.append({
            'component': component,
            'anemia_cases_millions': impacts[component]['anemia_cases'] / 1e6,
            'anemia_std_millions': impacts[component]['anemia_std'] / 1e6,
            'cases_averted_millions': impacts[component]['cases_averted'] / 1e6,
            'averted_std_millions': impacts[component]['averted_std'] / 1e6,
            'percent_reduction': impacts[component]['pct_reduction'],
        })

    df = pd.DataFrame(summary_data)
    df.to_csv('results/component_impacts_summary.csv', index=False)
    print('Saved impact summary to results/component_impacts_summary.csv')

    print('\n' + '='*80)
    print('ANALYSIS COMPLETE!')
    print('='*80)
    print('\nOutputs:')
    print('  - Figure: figures/component_impacts.png')
    print('  - Data:   results/component_impacts_summary.csv')
    print('  - Raw:    results/component_*.obj')
    print()


if __name__ == '__main__':
    main()
