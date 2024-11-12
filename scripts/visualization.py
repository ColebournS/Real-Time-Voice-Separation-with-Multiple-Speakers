import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import numpy as np

def visualize_results(df, results_dir, chunk_metrics=None):
    """
    Create comprehensive visualizations of the window and overlap analysis results,
    with consistent 3D surface plots for both overall and chunk-based metrics.
    
    Args:
        df: DataFrame with overall metrics
        results_dir: Directory to save visualizations
        chunk_metrics: Optional DataFrame with chunk-based metrics
    """
    sns.set_style("whitegrid")
    
    # Define metrics and their display names
    metrics = {
        'overall': {
            'sdr': 'SDR (dB)', 
            'sir': 'SIR (dB)', 
            'sar': 'SAR (dB)', 
            'total_separation_time': 'Processing Time (s)'
        },
        'chunk': {
            'chunk_average_sdr': 'Chunk Average SDR (dB)',
            'chunk_average_sir': 'Chunk Average SIR (dB)',
            'chunk_average_sar': 'Chunk Average SAR (dB)'
        }
    }
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        # Create 3D surface plots for overall metrics
        for metric, metric_name in metrics['overall'].items():
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            X = model_data['window_duration'].values
            Y = model_data['overlap_ratio'].values
            Z = model_data[metric].values
            
            try:
                surf = ax.plot_trisurf(X, Y, Z, 
                                     cmap='viridis' if metric == 'total_separation_time' else 'RdYlBu',
                                     linewidth=0.2, 
                                     antialiased=True)
                
                ax.set_title(f'3D Surface Plot of {metric_name} for {model}', 
                           size=14, pad=20)
                ax.set_xlabel('Window Duration (s)', size=12, labelpad=10)
                ax.set_ylabel('Overlap Ratio', size=12, labelpad=10)
                ax.set_zlabel(metric_name, size=12, labelpad=10)
                
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                cbar = plt.colorbar(surf, ax=ax, pad=0.1)
                cbar.ax.tick_params(labelsize=10)
                cbar.set_label(metric_name, size=12)
                
                # Rotate view for better visualization
                ax.view_init(elev=30, azim=45)
                
                plt.savefig(os.path.join(results_dir, f'surface_plot_{model}_{metric}.png'),
                           dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"Error creating/saving surface plot for {model}, metric '{metric}': {e}")
            finally:
                plt.close()
        
        # Create 3D surface plots for chunk-based metrics if available
        if chunk_metrics is not None:
            # Calculate average chunk metrics for each window/overlap combination
            chunk_averages = []
            for window in model_data['window_duration'].unique():
                for overlap in model_data['overlap_ratio'].unique():
                    chunk_data = chunk_metrics[
                        (chunk_metrics['model'] == model) &
                        (chunk_metrics['window_duration'] == window) &
                        (chunk_metrics['overlap_ratio'] == overlap)
                    ]
                    
                    if not chunk_data.empty:
                        avg_metrics = {
                            'window_duration': window,
                            'overlap_ratio': overlap,
                            'chunk_average_sdr': float(chunk_data['chunk_sdr'].mean()),
                            'chunk_average_sir': float(chunk_data['chunk_sir'].mean()),
                            'chunk_average_sar': float(chunk_data['chunk_sar'].mean())
                        }
                        chunk_averages.append(avg_metrics)
            
            if chunk_averages:
                # Convert to numpy arrays for plotting
                chunk_data_array = np.array([(d['window_duration'], d['overlap_ratio']) 
                                           for d in chunk_averages])
                
                # Create surface plots for chunk metrics
                for metric, metric_name in metrics['chunk'].items():
                    fig = plt.figure(figsize=(16, 12))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    Z = np.array([d[metric] for d in chunk_averages])
                    
                    try:
                        surf = ax.plot_trisurf(chunk_data_array[:, 0],
                                             chunk_data_array[:, 1],
                                             Z,
                                             cmap='RdYlBu',
                                             linewidth=0.2,
                                             antialiased=True)
                        
                        ax.set_title(f'3D Surface Plot of {metric_name} for {model}',
                                   size=14, pad=20)
                        ax.set_xlabel('Window Duration (s)', size=12, labelpad=10)
                        ax.set_ylabel('Overlap Ratio', size=12, labelpad=10)
                        ax.set_zlabel(metric_name, size=12, labelpad=10)
                        
                        ax.tick_params(axis='both', which='major', labelsize=10)
                        
                        cbar = plt.colorbar(surf, ax=ax, pad=0.1)
                        cbar.ax.tick_params(labelsize=10)
                        cbar.set_label(metric_name, size=12)
                        
                        # Rotate view for better visualization
                        ax.view_init(elev=30, azim=45)
                        
                        plt.savefig(os.path.join(results_dir, f'surface_plot_{model}_{metric}.png'),
                                  dpi=300, bbox_inches='tight')
                    except Exception as e:
                        print(f"Error creating/saving chunk surface plot for {model}, metric '{metric}': {e}")
                    finally:
                        plt.close()

def get_and_print_optimal_params(df, chunk_metrics=None):
    """
    Analyze and print optimal parameters, including chunk-based metrics if available.
    """
    try:
        optimal_params = find_optimal_parameters(df)
        
        print("\nAnalysis Summary:")
        for model, results in optimal_params.items():
            print(f"\nModel: {model}")
            
            for criterion, result in results.items():
                params = result['parameters']
                print(f"\n{result['criterion']}:")
                print(f"Window Duration: {params['window_duration']:.1f}s")
                print(f"Overlap Ratio: {params['overlap_ratio']:.1%}")
                print(f"Overall SDR: {params['sdr']:.2f} dB")
                print(f"Processing Time: {params['total_separation_time']:.2f}s")
                
                # Print chunk-based metrics if available
                if chunk_metrics is not None:
                    chunk_data = chunk_metrics[
                        (chunk_metrics['model'] == model) &
                        (chunk_metrics['window_duration'] == params['window_duration']) &
                        (chunk_metrics['overlap_ratio'] == params['overlap_ratio'])
                    ]
                    
                    if not chunk_data.empty:
                        print("\nChunk-based Metrics:")
                        for metric in ['sdr', 'sir', 'sar']:
                            values = chunk_data[f'chunk_{metric}']
                            print(f"Chunk {metric.upper()}:")
                            print(f"  Mean: {values.mean():.2f} dB")
                            print(f"  Std: {values.std():.2f} dB")
                            print(f"  Min: {values.min():.2f} dB")
                            print(f"  Max: {values.max():.2f} dB")
    
    except Exception as e:
        print(f"Error in finding optimal parameters: {e}")


def find_optimal_parameters(df):
    """
    Find optimal parameters based on different criteria.
    """
    optimal_results = {}
    
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        
        optimal_results[model] = {
            'best_quality': {
                'parameters': model_data.loc[model_data['sdr'].idxmax()],
                'criterion': 'Highest SDR'
            },
            'best_efficiency': {
                'parameters': model_data.loc[model_data['sdr'].div(model_data['total_separation_time']).idxmax()],
                'criterion': 'Best SDR/Time ratio'
            },
            'fastest': {
                'parameters': model_data.loc[model_data['total_separation_time'].idxmin()],
                'criterion': 'Fastest processing time'
            }
        }
    
    return optimal_results