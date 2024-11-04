import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from analysis import find_optimal_parameters

def visualize_results(df, results_dir):
    """
    Create comprehensive visualizations of the window and overlap analysis results.
    """
    # Set the style using seaborn's set_style instead of plt.style.use
    sns.set_style("whitegrid")
    
    # Define metrics and their display names
    metrics = ['sdr', 'sir', 'sar', 'total_separation_time']
    metric_names = {
        'sdr': 'SDR (dB)', 
        'sir': 'SIR (dB)', 
        'sar': 'SAR (dB)', 
        'total_separation_time': 'Processing Time (s)'
    }
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        # Create figure with subplots for each metric
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(2, 2, figure=fig)
        fig.suptitle(f'Performance Metrics for {model}', size=16, y=0.95)
        
        for idx, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[idx // 2, idx % 2])
            
            try:
                # Pivot data for heatmap
                heatmap_data = model_data.pivot(index='window_duration', 
                                                 columns='overlap_ratio',
                                                 values=metric)
                
                # Create heatmap with improved aesthetics
                sns.heatmap(heatmap_data, 
                            annot=True, 
                            fmt='.2f', 
                            cmap='viridis' if metric == 'total_separation_time' else 'RdYlBu',
                            ax=ax,
                            cbar_kws={'label': metric_names[metric]})
                
                ax.set_title(f'{metric_names[metric]} vs Window Duration and Overlap')
                ax.set_xlabel('Overlap Ratio')
                ax.set_ylabel('Window Duration (s)')
            except Exception as e:
                print(f"Error creating heatmap for {model}, metric '{metric}': {e}")
        
        plt.tight_layout()
        
        try:
            plt.savefig(os.path.join(results_dir, f'heatmap_analysis_{model}.png'), 
                        dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving heatmap for {model}: {e}")
        
        plt.close()
        
        # Create 3D surface plots with improved aesthetics
        for metric in metrics:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            X = model_data['window_duration'].values
            Y = model_data['overlap_ratio'].values
            Z = model_data[metric].values
            
            try:
                ax.plot_trisurf(X, Y, Z, cmap='viridis' if metric == 'total_separation_time' else 'RdYlBu', 
                                 linewidth=0.2, antialiased=True)
                
                ax.set_title(f'3D Surface Plot of {metric_names[metric]}')
                ax.set_xlabel('Window Duration (s)')
                ax.set_ylabel('Overlap Ratio')
                ax.set_zlabel(metric_names[metric])
                
                plt.savefig(os.path.join(results_dir, f'surface_plot_{model}_{metric}.png'), 
                            dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"Error creating/saving surface plot for {model}, metric '{metric}': {e}")
            finally:
                plt.close()

def get_and_print_optimal_params(df):
    try:
        optimal_params = find_optimal_parameters(df)
        
        # Print summary and recommendations
        print("\nAnalysis Summary:")
        for model, results in optimal_params.items():
            print(f"\nModel: {model}")
            for criterion, result in results.items():
                params = result['parameters']
                print(f"\n{result['criterion']}:")
                print(f"Window Duration: {params['window_duration']:.1f}s")
                print(f"Overlap Ratio: {params['overlap_ratio']:.1%}")
                print(f"SDR: {params['sdr']:.2f} dB")
                print(f"Processing Time: {params['total_separation_time']:.2f}s")
    except Exception as e:
        print(f"Error in finding optimal parameters: {e}")
