import os
import pandas as pd
import numpy as np

def convert_phase0_csv_to_npz(csv_path, output_npz_path, du_names, is_test=False):
    """
    Converts experimental_phase0 CSV files (with focal_ and sib_ prefixes)
    into the multi-dimensional .npz tensors expected by TopoAR.
    """
    if not os.path.exists(csv_path):
        print(f"[-] File not found: {csv_path}")
        return

    # 1. Read the CSV file
    df = pd.read_csv(csv_path)
    time_steps = len(df)
    num_dus = len(du_names)
    
    print(f"[*] Converting {csv_path}...")
    print(f"    -> Time steps: {time_steps} | Detected DUs: {num_dus}")

    # 2. Extract Central Unit (CU) Features (Shape: Time_Steps, 5)
    cu_features = [
        'cu_cpu_pct', 'cu_mem_bytes', 'cu_net_tx_per_du', 
        'cu_net_rx', 'cu_fs_read'
    ]
    # Fallback to zeros if any metrics are missing from a particular run
    for col in cu_features:
        if col not in df.columns:
            df[col] = 0.0
            
    cu_matrix = df[cu_features].to_numpy(dtype=np.float32)

    # 3. Build the 3D Distributed Unit (DU) Tensor 
    # TopoAR expects Shape: (Time_Steps, Number_of_DUs, 4_Features)
    # Feature mapping matching the repo: [cpu_pct, mem_bytes, fs_write, fs_read]
    du_base_features = ['du_cpu_pct', 'du_mem_bytes', 'du_fs_write', 'du_fs_read']
    du_tensor = np.zeros((time_steps, num_dus, len(du_base_features)), dtype=np.float32)

    for du_idx, du_name in enumerate(du_names):
        # Phase 0 maps the first DU as 'focal_' and subsequent DUs as 'sib_'
        prefix = "focal_" if du_idx == 0 else "sib_"
        
        for feat_idx, base_feat in enumerate(du_base_features):
            col_name = f"{prefix}{base_feat}"
            
            if col_name in df.columns:
                du_tensor[:, du_idx, feat_idx] = df[col_name].to_numpy()
            else:
                # If a specific sibling feature is missing, keep it 0.0
                du_tensor[:, du_idx, feat_idx] = 0.0

    # 4. Extract or Generate Labels (0 = Normal, 1 = Anomaly)
    if is_test and 'label' in df.columns:
        labels = df['label'].to_numpy(dtype=np.int32)
    elif is_test and 'anomaly' in df.columns:
        labels = df['anomaly'].to_numpy(dtype=np.int32)
    else:
        # Default to healthy 0s for training sets
        labels = np.zeros(time_steps, dtype=np.int32)

    # 5. Save directly into the NumPy zipped archive format
    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    np.savez(
        output_npz_path,
        cu=cu_matrix,
        du=du_tensor,
        label=labels,
        du_names=np.array(du_names, dtype=object)
    )
    print(f"[+] Successfully generated: {output_npz_path}\n")


if __name__ == "__main__":
    # --- SAVE FILES DIRECTLY IN THE CURRENT DIRECTORY ---
    CURRENT_DIR = "."

    # 1. Convert your local train_pairs.csv -> ./train.npz
    convert_phase0_csv_to_npz(
        csv_path="train_pairs.csv",
        output_npz_path=f"{CURRENT_DIR}/train.npz",
        du_names=["srsdu0", "srsdu1"],
        is_test=False
    )
    
    # 2. Convert your local test_pairs.csv -> ./test.npz
    convert_phase0_csv_to_npz(
        csv_path="test_pairs.csv",
        output_npz_path=f"{CURRENT_DIR}/test.npz",
        du_names=["srsdu0", "srsdu1"],
        is_test=True
    )
