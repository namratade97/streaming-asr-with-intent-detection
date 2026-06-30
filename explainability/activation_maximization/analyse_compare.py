import pandas as pd
import numpy as np

def analyze_phonemic_activations(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Use absolute magnitude 
    df['abs_activation'] = df['activation'].abs()
    
    audios = df['audio'].unique()
    layers = df['layer'].unique()
    
    print(f"Found audios: {list(audios)}")
    print(f"Found {len(layers)} structural layers.\n")
    
    # Filter out dead/inactive neurons
    active_neurons_df = df[df['abs_activation'] > 1e-9]
    
    top_n = 10
    
    # Sort descending and group by audio and layer to fetch the top 10 actual firing neurons
    top_neurons = (active_neurons_df.sort_values(by='abs_activation', ascending=False)
                   .groupby(['audio', 'layer'])
                   .head(top_n))
    
    fingerprint_records = []
    anchor_records = []
    
    for layer in layers:
        layer_data = top_neurons[top_neurons['layer'] == layer]
        
        # Create sets of top neurons for each audio in this specific layer
        neuron_sets = {}
        for audio in audios:
            neuron_sets[audio] = set(layer_data[layer_data['audio'] == audio]['neuron'])
            
        # Find Acoustic Anchors (Intersect of all sets in this layer)
        all_sets = list(neuron_sets.values())
        # Check if we actually have data for all 3 audios in this layer after filtering zeros
        if len(neuron_sets) == len(audios) and all(len(s) > 0 for s in all_sets):
            anchors = set.intersection(*all_sets)
            for neuron in anchors:
                anchor_records.append({'layer': layer, 'neuron': neuron})

        # Find Phonetic Fingerprints (Present in one audio's top set, but none of the others)
        for audio in audios:
            if audio in neuron_sets:
                other_neurons = set()
                for other_audio in audios:
                    if other_audio != audio and other_audio in neuron_sets:
                        other_neurons.update(neuron_sets[other_audio])
                
                unique_to_audio = neuron_sets[audio] - other_neurons
                for neuron in unique_to_audio:
                    fingerprint_records.append({
                        'audio': audio,
                        'layer': layer,
                        'neuron': neuron
                    })
                
    # Convert results to DataFrames
    df_fingerprints = pd.DataFrame(fingerprint_records)
    df_anchors = pd.DataFrame(anchor_records)
    
    print("=" * 60)
    print("QUANTITATIVE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total Unique 'Phonetic Fingerprint' Neurons Found: {len(df_fingerprints)}")
    print(f"Total Shared 'Acoustic Anchor' Neurons Found: {len(df_anchors)}")
    print("-" * 60)
    
    print("\nBreakdown of Unique 'Phonetic Fingerprint' Neurons per Audio:")
    if not df_fingerprints.empty:
        print(df_fingerprints['audio'].value_counts())
    else:
        print("None found.")
        
    print("\nSample of Phonetic Fingerprint Neurons:")
    if not df_fingerprints.empty:
        print(df_fingerprints.head(15).to_string(index=False))
        
    print("\nSample of Acoustic Anchor Neurons (Shared across all profiles):")
    if not df_anchors.empty:
        print(df_anchors.head(15).to_string(index=False))
        
    df_fingerprints.to_csv('/Users/nde/Downloads/streaming-asr-with-intent-detection-results/phonetic_fingerprints.csv', index=False)
    df_anchors.to_csv('/Users/nde/Downloads/streaming-asr-with-intent-detection-results/acoustic_anchors.csv', index=False)
    print("\nResults exported successfully to 'phonetic_fingerprints.csv' and 'acoustic_anchors.csv'.")

if __name__ == "__main__":
    analyze_phonemic_activations('/Users/nde/Downloads/streaming-asr-with-intent-detection-results/phoneme_layerwise_activations_2/layerwise_mean_activations_2.csv')