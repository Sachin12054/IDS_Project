"""
================================================================================
RUN ALL ANALYSES - MASTER SCRIPT
================================================================================
Executes all visualization and analysis scripts in sequence:
1. Time Series Models (SARIMA, XGBoost, LSTM)
2. Advanced Time Series Analysis (Spectral, Granger, etc.)
3. Enhanced Visualizations (Model Comparison, Heatmaps)

Author: Computer Security Project - Amrita University
Date: January 2026
================================================================================
"""

import os
import sys
import time
from datetime import datetime

def print_banner(title):
    """Print a formatted banner"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def run_script(script_name, description):
    """Run a Python script and track execution"""
    print_banner(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start_time = time.time()
    
    try:
        # Execute the script
        exec(open(script_name).read())
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ {description} completed successfully!")
        print(f"⏱️  Execution time: {elapsed_time:.2f} seconds")
        return True, elapsed_time
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Error in {description}:")
        print(f"   {str(e)}")
        print(f"⏱️  Time before error: {elapsed_time:.2f} seconds")
        return False, elapsed_time

def main():
    """Main execution function"""
    print_banner("IDS PROJECT - COMPREHENSIVE ANALYSIS SUITE")
    print("This script will execute all visualization and analysis components")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get the notebooks directory
    notebooks_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define scripts to run
    scripts = [
        (os.path.join(notebooks_dir, 'time_series_models.py'), 
         "Time Series Models (SARIMA, XGBoost, LSTM)"),
        (os.path.join(notebooks_dir, 'advanced_time_series_analysis.py'),
         "Advanced Time Series Analysis"),
        (os.path.join(notebooks_dir, 'enhanced_visualizations.py'),
         "Enhanced Visualizations & Model Comparison")
    ]
    
    results = []
    total_start = time.time()
    
    # Execute each script
    for script_path, description in scripts:
        if os.path.exists(script_path):
            success, exec_time = run_script(script_path, description)
            results.append({
                'script': description,
                'success': success,
                'time': exec_time
            })
        else:
            print(f"\n⚠️  Script not found: {script_path}")
            results.append({
                'script': description,
                'success': False,
                'time': 0
            })
    
    # Print summary
    total_time = time.time() - total_start
    
    print_banner("EXECUTION SUMMARY")
    
    print(f"{'Script':<50} {'Status':<15} {'Time (s)':<10}")
    print("-" * 80)
    
    success_count = 0
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{result['script']:<50} {status:<15} {result['time']:>8.2f}")
        if result['success']:
            success_count += 1
    
    print("-" * 80)
    print(f"\nTotal Scripts: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    print(f"\n⏱️  Total Execution Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"🏁 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List all generated outputs
    print_banner("GENERATED OUTPUTS")
    
    output_base = os.path.join(os.path.dirname(notebooks_dir), "evaluation_results")
    
    directories = [
        'time_series_models',
        'advanced_time_series', 
        'enhanced_visualizations',
        'time_series_analysis'
    ]
    
    for directory in directories:
        dir_path = os.path.join(output_base, directory)
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.md', '.txt', '.json'))]
            if files:
                print(f"\n📁 {directory}/")
                for file in sorted(files):
                    print(f"   - {file}")
    
    print("\n" + "=" * 80)
    print("  ALL ANALYSES COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
