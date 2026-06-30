import pandas as pd

def generate_anchor_validation(original_csv, anchors_csv, output_csv):
    print("Loading datasets...")
    df = pd.read_csv(original_csv)
    df_anchors = pd.read_csv(anchors_csv)
    
    # Calculate absolute activation for sorting/checking
    df['abs_activation'] = df['activation'].abs()
    
    validation_records = []
    
    print("Validating anchors...")
    for _, row in df_anchors.iterrows():
        layer_name = row['layer']
        neuron_idx = row['neuron']
        
        # Pull all data for this specific neuron from the original data
        neuron_data = df[(df['layer'] == layer_name) & (df['neuron'] == neuron_idx)]
        
        # Pivot the data to see the audios side-by-side
        record = {
            'layer': layer_name,
            'neuron': neuron_idx
        }
        
        # Add the raw activation values for each audio file
        for _, n_row in neuron_data.iterrows():
            record[f"{n_row['audio']}_val"] = n_row['activation']
            record[f"{n_row['audio']}_abs"] = n_row['abs_activation']
            
        validation_records.append(record)
        
    df_validation = pd.DataFrame(validation_records)
    
    df_validation.to_csv(output_csv, index=False)
    print(f"\nValidation ledger successfully saved to: {output_csv}")

if __name__ == "__main__":
    generate_anchor_validation(
        original_csv='/Users/nde/Downloads/streaming-asr-with-intent-detection-results/phoneme_layerwise_activations_2/layerwise_mean_activations_2.csv',
        anchors_csv='/Users/nde/Downloads/streaming-asr-with-intent-detection-results/acoustic_anchors.csv',
        output_csv='/Users/nde/Downloads/streaming-asr-with-intent-detection-results/anchor_validation_check.csv'
    )