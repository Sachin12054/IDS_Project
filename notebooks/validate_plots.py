"""
Validate all generated plots for correctness
"""
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

MISSING_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "missing")

def validate_images():
    """Validate all PNG files"""
    print("="*80)
    print("  VALIDATING ALL GENERATED PLOTS")
    print("="*80 + "\n")
    
    categories = {
        'time_series_models': [
            'arima_diagnostics.png',
            'error_distributions.png',
            'lstm_learning_curves.png',
            'model_comparison.png',
            'model_metrics_comparison.png',
            'prediction_intervals.png',
            'residual_analysis.png',
            'xgboost_feature_importance.png'
        ],
        'advanced_time_series': [
            'cross_correlation.png',
            'spectral_analysis.png',
            'structural_breaks.png'
        ],
        'enhanced_visualizations': [
            'attack_pattern_heatmaps.png',
            'comprehensive_model_comparison.png',
            'metric_evolution.png'
        ]
    }
    
    total_files = 0
    valid_files = 0
    invalid_files = []
    
    for category, files in categories.items():
        print(f"\n📁 {category}/")
        print("-" * 60)
        
        for filename in files:
            filepath = os.path.join(MISSING_DIR, category, filename)
            total_files += 1
            
            try:
                # Check if file exists
                if not os.path.exists(filepath):
                    print(f"  ❌ {filename}: FILE NOT FOUND")
                    invalid_files.append(f"{category}/{filename} - Not Found")
                    continue
                
                # Check file size
                file_size = os.path.getsize(filepath) / 1024  # KB
                if file_size < 10:
                    print(f"  ⚠️  {filename}: TOO SMALL ({file_size:.1f} KB)")
                    invalid_files.append(f"{category}/{filename} - Too Small")
                    continue
                
                # Try to open and validate as image
                img = Image.open(filepath)
                width, height = img.size
                
                # Check if image has reasonable dimensions
                if width < 100 or height < 100:
                    print(f"  ⚠️  {filename}: INVALID DIMENSIONS ({width}x{height})")
                    invalid_files.append(f"{category}/{filename} - Invalid Dimensions")
                    continue
                
                # Verify image mode
                if img.mode not in ['RGB', 'RGBA', 'L']:
                    print(f"  ⚠️  {filename}: INVALID MODE ({img.mode})")
                    invalid_files.append(f"{category}/{filename} - Invalid Mode")
                    continue
                
                print(f"  ✅ {filename}: OK ({file_size:.1f} KB, {width}x{height}, {img.mode})")
                valid_files += 1
                img.close()
                
            except Exception as e:
                print(f"  ❌ {filename}: ERROR - {str(e)}")
                invalid_files.append(f"{category}/{filename} - {str(e)}")
    
    # Check text report
    print(f"\n📁 advanced_time_series/")
    print("-" * 60)
    granger_file = os.path.join(MISSING_DIR, "advanced_time_series", "granger_causality_results.txt")
    if os.path.exists(granger_file):
        file_size = os.path.getsize(granger_file)
        if file_size > 0:
            print(f"  ✅ granger_causality_results.txt: OK ({file_size} bytes)")
            valid_files += 1
            total_files += 1
        else:
            print(f"  ❌ granger_causality_results.txt: EMPTY")
            invalid_files.append("advanced_time_series/granger_causality_results.txt - Empty")
    else:
        print(f"  ❌ granger_causality_results.txt: NOT FOUND")
        invalid_files.append("advanced_time_series/granger_causality_results.txt - Not Found")
    
    # Summary
    print("\n" + "="*80)
    print("  VALIDATION SUMMARY")
    print("="*80)
    print(f"\nTotal Files Expected: {total_files}")
    print(f"Valid Files: {valid_files}")
    print(f"Invalid/Missing Files: {len(invalid_files)}")
    
    if invalid_files:
        print("\n⚠️  ISSUES FOUND:")
        for issue in invalid_files:
            print(f"  - {issue}")
    else:
        print("\n✅ ALL PLOTS ARE VALID!")
    
    print("\n" + "="*80 + "\n")
    
    return valid_files == total_files

def display_sample_plots():
    """Display sample plots for visual verification"""
    print("\n" + "="*80)
    print("  SAMPLE PLOT PREVIEW")
    print("="*80 + "\n")
    
    sample_plots = [
        ('time_series_models', 'model_comparison.png', 'Time Series Model Comparison'),
        ('enhanced_visualizations', 'comprehensive_model_comparison.png', 'Comprehensive Comparison'),
        ('advanced_time_series', 'spectral_analysis.png', 'Spectral Analysis')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, (category, filename, title) in enumerate(sample_plots):
        filepath = os.path.join(MISSING_DIR, category, filename)
        try:
            img = mpimg.imread(filepath)
            axes[idx].imshow(img)
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].axis('off')
            print(f"  ✅ Loaded: {category}/{filename}")
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'ERROR\n{str(e)}', 
                          ha='center', va='center', fontsize=12)
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].axis('off')
            print(f"  ❌ Error loading: {category}/{filename}")
    
    plt.tight_layout()
    preview_path = os.path.join(MISSING_DIR, 'validation_preview.png')
    plt.savefig(preview_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Sample preview saved: {preview_path}\n")
    print("="*80 + "\n")

if __name__ == "__main__":
    all_valid = validate_images()
    
    if all_valid:
        print("🎉 All plots validated successfully!")
        try:
            display_sample_plots()
        except Exception as e:
            print(f"⚠️  Could not generate preview: {e}")
    else:
        print("⚠️  Some plots have issues. Please check the log above.")
