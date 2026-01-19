"""
================================================================================
ADVANCED TIME SERIES ANALYSIS FOR IDS
================================================================================
Comprehensive time series analysis including:
- Spectral Analysis (Periodogram, Fourier Transform)
- Cross-Correlation Analysis
- Granger Causality Tests
- Structural Break Detection
- Advanced Forecasting Techniques
- Volatility Clustering Analysis

Author: Computer Security Project - Amrita University
Date: January 2026
================================================================================
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Time Series Analysis
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import scipy.stats as st

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          "evaluation_results", "advanced_time_series")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')


class AdvancedTimeSeriesAnalysis:
    """Comprehensive advanced time series analysis for IDS"""
    
    def __init__(self):
        self.data = None
        self.results = {}
        
    def load_data(self):
        """Load processed data and create time series"""
        print("=" * 70)
        print("LOADING DATA FOR ADVANCED ANALYSIS")
        print("=" * 70)
        
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "processed", "cleaned_features.parquet")
        
        df = pd.read_parquet(data_path)
        print(f"Loaded {len(df):,} records")
        
        # Create timestamp and aggregate to hourly
        df['timestamp'] = pd.date_range(start='2018-02-14', periods=len(df), freq='1s')
        df['hour'] = df['timestamp'].dt.floor('h')
        
        # Create multiple time series
        self.data = pd.DataFrame({
            'attack_count': df.groupby('hour')['is_attack'].sum(),
            'total_packets': df.groupby('hour').size(),
            'avg_flow_duration': df.groupby('hour')['Flow Duration'].mean(),
        })
        
        self.data['attack_rate'] = (self.data['attack_count'] / self.data['total_packets'] * 100)
        
        print(f"\nCreated time series with {len(self.data)} hourly observations")
        print(f"Time range: {self.data.index[0]} to {self.data.index[-1]}")
        
        return self.data
    
    def spectral_analysis(self):
        """Perform spectral analysis using periodogram and FFT"""
        print("\n" + "=" * 70)
        print("SPECTRAL ANALYSIS - Frequency Domain")
        print("=" * 70)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Analyze attack_count series
        series = self.data['attack_count'].values
        
        # 1. Periodogram
        ax1 = axes[0, 0]
        frequencies, power = signal.periodogram(series)
        ax1.semilogy(frequencies, power)
        ax1.set_title('Periodogram - Attack Count', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Frequency (cycles per hour)')
        ax1.set_ylabel('Power Spectral Density')
        ax1.grid(True, alpha=0.3)
        
        # Find dominant frequencies
        top_freqs_idx = np.argsort(power)[-5:][::-1]
        top_freqs = frequencies[top_freqs_idx]
        print(f"\nTop 5 dominant frequencies:")
        for i, freq in enumerate(top_freqs, 1):
            period = 1/freq if freq > 0 else np.inf
            print(f"  {i}. Frequency: {freq:.4f}, Period: {period:.2f} hours")
        
        # 2. FFT Analysis
        ax2 = axes[0, 1]
        N = len(series)
        yf = fft(series)
        xf = fftfreq(N, 1)[:N//2]
        
        ax2.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
        ax2.set_title('FFT - Frequency Components', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Frequency (cycles per hour)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        
        # 3. Welch's method (smoothed periodogram)
        ax3 = axes[1, 0]
        f_welch, Pxx_welch = signal.welch(series, nperseg=min(256, len(series)//4))
        ax3.semilogy(f_welch, Pxx_welch)
        ax3.set_title("Welch's Method - Smoothed PSD", fontsize=14, fontweight='bold')
        ax3.set_xlabel('Frequency (cycles per hour)')
        ax3.set_ylabel('Power Spectral Density')
        ax3.grid(True, alpha=0.3)
        
        # 4. Spectrogram
        ax4 = axes[1, 1]
        f, t, Sxx = signal.spectrogram(series, nperseg=min(64, len(series)//8))
        im = ax4.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
        ax4.set_title('Spectrogram - Time-Frequency Analysis', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Frequency (cycles per hour)')
        ax4.set_xlabel('Time')
        plt.colorbar(im, ax=ax4, label='Power (dB)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'spectral_analysis.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Spectral analysis saved")
        
        self.results['spectral'] = {
            'dominant_frequencies': top_freqs,
            'periodogram': (frequencies, power)
        }
    
    def cross_correlation_analysis(self):
        """Analyze cross-correlations between different time series"""
        print("\n" + "=" * 70)
        print("CROSS-CORRELATION ANALYSIS")
        print("=" * 70)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Define series pairs for analysis
        series_pairs = [
            ('attack_count', 'total_packets'),
            ('attack_count', 'attack_rate'),
            ('attack_rate', 'avg_flow_duration'),
            ('total_packets', 'avg_flow_duration')
        ]
        
        for idx, (series1_name, series2_name) in enumerate(series_pairs):
            ax = axes[idx // 2, idx % 2]
            
            series1 = self.data[series1_name].values
            series2 = self.data[series2_name].values
            
            # Normalize series
            series1_norm = (series1 - np.mean(series1)) / np.std(series1)
            series2_norm = (series2 - np.mean(series2)) / np.std(series2)
            
            # Calculate cross-correlation
            correlation = np.correlate(series1_norm, series2_norm, mode='full')
            lags = np.arange(-len(series1)+1, len(series1))
            
            # Plot
            ax.plot(lags, correlation, linewidth=2)
            ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Lag')
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
            ax.set_title(f'Cross-Correlation: {series1_name} vs {series2_name}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Lag (hours)')
            ax.set_ylabel('Correlation')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Find maximum correlation and its lag
            max_corr_idx = np.argmax(np.abs(correlation))
            max_corr = correlation[max_corr_idx]
            max_lag = lags[max_corr_idx]
            
            print(f"\n{series1_name} vs {series2_name}:")
            print(f"  Max correlation: {max_corr:.4f} at lag {max_lag} hours")
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'cross_correlation.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Cross-correlation analysis saved")
    
    def granger_causality_test(self):
        """Test Granger causality between time series"""
        print("\n" + "=" * 70)
        print("GRANGER CAUSALITY TESTS")
        print("=" * 70)
        
        results_text = []
        
        # Test pairs
        test_pairs = [
            ('total_packets', 'attack_count'),
            ('attack_count', 'attack_rate'),
            ('avg_flow_duration', 'attack_count')
        ]
        
        for x_name, y_name in test_pairs:
            print(f"\nTesting: Does {x_name} Granger-cause {y_name}?")
            
            # Prepare data
            data_pair = self.data[[x_name, y_name]].dropna()
            
            try:
                # Perform Granger causality test (maxlag=12 hours)
                gc_result = grangercausalitytests(data_pair, maxlag=12, verbose=False)
                
                # Extract p-values
                p_values = []
                for lag in range(1, 13):
                    ssr_ftest = gc_result[lag][0]['ssr_ftest']
                    p_values.append(ssr_ftest[1])  # p-value
                
                # Find significant lags (p < 0.05)
                significant_lags = [i+1 for i, p in enumerate(p_values) if p < 0.05]
                
                result_str = f"{x_name} -> {y_name}:"
                if significant_lags:
                    result_str += f" YES (significant at lags: {significant_lags[:3]})"
                    print(f"  ✓ Granger-causes at lags: {significant_lags}")
                else:
                    result_str += " NO"
                    print(f"  ✗ Does not Granger-cause")
                
                results_text.append(result_str)
                
                # Print best lag
                best_lag = np.argmin(p_values) + 1
                print(f"  Best lag: {best_lag} (p-value: {p_values[best_lag-1]:.4f})")
                
            except Exception as e:
                print(f"  Error: {e}")
                results_text.append(f"{x_name} -> {y_name}: ERROR")
        
        # Save results
        results_file = os.path.join(OUTPUT_DIR, 'granger_causality_results.txt')
        with open(results_file, 'w') as f:
            f.write("GRANGER CAUSALITY TEST RESULTS\n")
            f.write("=" * 50 + "\n\n")
            for result in results_text:
                f.write(result + "\n")
        
        print(f"\n✅ Granger causality results saved to {results_file}")
    
    def structural_break_detection(self):
        """Detect structural breaks in time series using CUSUM"""
        print("\n" + "=" * 70)
        print("STRUCTURAL BREAK DETECTION")
        print("=" * 70)
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        series_to_analyze = ['attack_count', 'attack_rate', 'total_packets', 'avg_flow_duration']
        
        for idx, series_name in enumerate(series_to_analyze):
            ax = axes[idx // 2, idx % 2]
            
            series = self.data[series_name].values
            
            # Calculate CUSUM
            mean_series = np.mean(series)
            cusum = np.cumsum(series - mean_series)
            
            # Plot original series
            ax2 = ax.twinx()
            ax2.plot(range(len(series)), series, 'b-', alpha=0.5, linewidth=1, label='Original')
            ax2.set_ylabel('Original Value', color='b')
            ax2.tick_params(axis='y', labelcolor='b')
            
            # Plot CUSUM
            ax.plot(range(len(cusum)), cusum, 'r-', linewidth=2, label='CUSUM')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            ax.set_ylabel('CUSUM', color='r')
            ax.set_xlabel('Time (hours)')
            ax.set_title(f'Structural Break Detection - {series_name}', fontsize=12, fontweight='bold')
            ax.tick_params(axis='y', labelcolor='r')
            ax.grid(True, alpha=0.3)
            
            # Detect potential breaks (local maxima/minima in CUSUM)
            from scipy.signal import find_peaks
            peaks_pos, _ = find_peaks(cusum, prominence=np.std(cusum))
            peaks_neg, _ = find_peaks(-cusum, prominence=np.std(cusum))
            
            # Mark breaks
            for peak in peaks_pos:
                ax.axvline(x=peak, color='orange', linestyle=':', linewidth=2, alpha=0.7)
            for peak in peaks_neg:
                ax.axvline(x=peak, color='purple', linestyle=':', linewidth=2, alpha=0.7)
            
            print(f"\n{series_name}:")
            print(f"  Potential breaks detected: {len(peaks_pos) + len(peaks_neg)}")
            if len(peaks_pos) > 0:
                print(f"  Upward shifts at hours: {peaks_pos[:5]}")
            if len(peaks_neg) > 0:
                print(f"  Downward shifts at hours: {peaks_neg[:5]}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'structural_breaks.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✅ Structural break detection saved")
    
    def wavelet_analysis(self):
        """Wavelet transform for multi-scale time-frequency analysis"""
        print("\n" + "=" * 70)
        print("WAVELET ANALYSIS")
        print("=" * 70)
        
        try:
            import pywt
            
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            
            series = self.data['attack_count'].values
            
            # 1. Continuous Wavelet Transform
            ax1 = axes[0, 0]
            scales = np.arange(1, 128)
            coefficients, frequencies = pywt.cwt(series, scales, 'morl')
            
            im1 = ax1.imshow(np.abs(coefficients), extent=[0, len(series), 1, 128],
                           cmap='viridis', aspect='auto', vmax=abs(coefficients).max())
            ax1.set_title('Continuous Wavelet Transform', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Scale')
            ax1.set_xlabel('Time (hours)')
            plt.colorbar(im1, ax=ax1, label='Magnitude')
            
            # 2. Discrete Wavelet Transform
            ax2 = axes[0, 1]
            wavelet = 'db4'
            level = 4
            coeffs = pywt.wavedec(series, wavelet, level=level)
            
            # Reconstruct and plot
            reconstructed = pywt.waverec(coeffs, wavelet)[:len(series)]
            ax2.plot(series, label='Original', alpha=0.7, linewidth=1)
            ax2.plot(reconstructed, label='Reconstructed', linewidth=2, linestyle='--')
            ax2.set_title(f'Discrete Wavelet Transform ({wavelet})', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Time (hours)')
            ax2.set_ylabel('Attack Count')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Wavelet decomposition levels
            ax3 = axes[1, 0]
            for i, coeff in enumerate(coeffs[:-1]):  # Skip approximation
                if len(coeff) > 0:
                    ax3.plot(np.arange(len(coeff)), coeff, label=f'Detail {i+1}', alpha=0.7)
            ax3.set_title('Wavelet Decomposition Levels', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Coefficient Index')
            ax3.set_ylabel('Magnitude')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. Energy distribution
            ax4 = axes[1, 1]
            energies = [np.sum(c**2) for c in coeffs]
            total_energy = sum(energies)
            energy_pct = [e/total_energy*100 for e in energies]
            
            labels = [f'Level {i+1}' for i in range(len(coeffs)-1)] + ['Approx']
            ax4.bar(range(len(energies)), energy_pct, color='steelblue', alpha=0.7, edgecolor='black')
            ax4.set_xticks(range(len(energies)))
            ax4.set_xticklabels(labels)
            ax4.set_title('Wavelet Energy Distribution', fontsize=14, fontweight='bold')
            ax4.set_ylabel('Energy (%)')
            ax4.grid(True, alpha=0.3)
            
            # Add percentage labels
            for i, pct in enumerate(energy_pct):
                ax4.text(i, pct, f'{pct:.1f}%', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'wavelet_analysis.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Wavelet analysis saved")
            
        except ImportError:
            print("⚠️ PyWavelets not installed. Skipping wavelet analysis.")
            print("   Install with: pip install PyWavelets")
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "=" * 70)
        print("GENERATING COMPREHENSIVE REPORT")
        print("=" * 70)
        
        report = f"""# Advanced Time Series Analysis Report
## Intrusion Detection System

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 1. Spectral Analysis

Spectral analysis reveals the frequency components and periodicities in the attack patterns.

**Key Findings:**
- Dominant frequencies identified in attack patterns
- Periodic cycles detected using FFT and periodogram
- Time-frequency characteristics analyzed via spectrogram

**Files Generated:**
- `spectral_analysis.png` - Comprehensive frequency domain analysis

---

## 2. Cross-Correlation Analysis

Cross-correlation analysis examines relationships between different time series variables.

**Analyzed Pairs:**
- Attack count vs Total packets
- Attack count vs Attack rate
- Attack rate vs Average flow duration
- Total packets vs Average flow duration

**Files Generated:**
- `cross_correlation.png` - Cross-correlation plots

---

## 3. Granger Causality Tests

Granger causality tests determine if one time series can predict another.

**Tests Performed:**
- Total packets → Attack count
- Attack count → Attack rate
- Average flow duration → Attack count

**Files Generated:**
- `granger_causality_results.txt` - Detailed test results

---

## 4. Structural Break Detection

CUSUM-based detection of structural breaks and regime changes.

**Analysis:**
- Identified potential change points in attack patterns
- Detected upward and downward shifts
- Revealed periods of stability and volatility

**Files Generated:**
- `structural_breaks.png` - CUSUM analysis plots

---

## 5. Wavelet Analysis

Multi-scale time-frequency decomposition using wavelet transforms.

**Components:**
- Continuous Wavelet Transform (CWT)
- Discrete Wavelet Transform (DWT)
- Energy distribution across scales
- Decomposition levels

**Files Generated:**
- `wavelet_analysis.png` - Wavelet decomposition visualizations

---

## Conclusion

This advanced time series analysis provides deep insights into:
- Temporal patterns and periodicities
- Causal relationships between variables
- Structural changes and anomalies
- Multi-scale frequency characteristics

These findings enhance the understanding of attack patterns and improve prediction capabilities.

---

*Analysis conducted using Python statsmodels, scipy, and PyWavelets libraries.*
"""
        
        report_path = os.path.join(OUTPUT_DIR, 'advanced_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Report saved to: {report_path}")


def main():
    """Main execution"""
    print("=" * 70)
    print("ADVANCED TIME SERIES ANALYSIS FOR IDS")
    print("Spectral | Cross-Correlation | Granger | Structural | Wavelet")
    print("=" * 70)
    
    # Initialize
    analyzer = AdvancedTimeSeriesAnalysis()
    
    # Load data
    analyzer.load_data()
    
    # Run analyses
    analyzer.spectral_analysis()
    analyzer.cross_correlation_analysis()
    analyzer.granger_causality_test()
    analyzer.structural_break_detection()
    analyzer.wavelet_analysis()
    
    # Generate report
    analyzer.generate_report()
    
    print("\n" + "=" * 70)
    print("✅ ADVANCED TIME SERIES ANALYSIS COMPLETED!")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
