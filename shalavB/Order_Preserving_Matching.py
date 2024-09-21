import os
import time
from pathlib import Path
import scipy.io as sio
import numpy as np
from Bio.Align import PairwiseAligner

# Define directories directly
adhd_dir = Path("Clean_DB/Clean_ADHD_DB")
control_dir = Path("Clean_DB/Clean_Control_DB")

# Output directories for raw data
adhd_output_dir = Path("Order_Preserving_Matching/ADHD_Raw_Data_Sequences")
control_output_dir = Path("Order_Preserving_Matching/Control_Raw_Data_Sequences")

# Create output directories if they don't exist
adhd_output_dir.mkdir(parents=True, exist_ok=True)
control_output_dir.mkdir(parents=True, exist_ok=True)

# Downsampling rate
DOWNSAMPLE_RATE = 4  # Keep every 4th sample

def downsample(sequence, rate):
    return sequence[::rate]

def save_channels_to_file(eeg_data, output_path, filename, downsample_rate):
    num_channels = eeg_data.shape[0]

    # Prepare content to write to file
    file_content = []
    for i in range(num_channels):
        channel_data = downsample(eeg_data[i, :].astype(int), downsample_rate)  # Downsample and convert to integers

        # Convert data to a sequence of + and - signs
        channel_string = ''.join(f"+{val}" if val > 0 else f"{val}" for val in channel_data)

        file_content.append(f"Channel {i + 1}: {channel_string}")

    # Write content to a single file
    filepath = output_path / f"{filename}.txt"
    with open(filepath, 'w') as f:
        f.write('\n'.join(file_content))
    print(f"Saved {filepath}")

def process_single_file(input_file, output_dir, downsample_rate):
    print(f"Processing {input_file.name}")
    mat_data = sio.loadmat(input_file)

    # Extract the key that corresponds to the filename (excluding '.mat' extension)
    key = input_file.stem
    if key not in mat_data:
        print(f"Key '{key}' not found in {input_file.name}")
        return None

    eeg_data = mat_data[key]

    # Check if eeg_data is a numpy array and has the expected dimensions
    if isinstance(eeg_data, np.ndarray) and len(eeg_data.shape) == 2:
        save_channels_to_file(eeg_data, output_dir, key, downsample_rate)
        return key
    else:
        print(f"Unexpected data format in {input_file.name}")
        return None

def needleman_wunsch(seq1, seq2):
    aligner = PairwiseAligner()
    aligner.mode = 'global'  # Set to global alignment

    try:
        # Perform the alignment
        alignments = aligner.align(seq1, seq2)
        best_alignment = alignments[0]  # Get the best alignment

        # Extract the aligned sequences
        aligned_seq1 = str(best_alignment[0])
        aligned_seq2 = str(best_alignment[1])

        # Calculate percentage similarity
        matches = sum(a == b for a, b in zip(aligned_seq1, aligned_seq2))
        total = max(len(aligned_seq1), len(aligned_seq2))
        percentage = (matches / total) * 100 if total > 0 else 0.0
        return percentage
    except MemoryError:
        print("Memory error occurred during alignment.")
        return 0.0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0.0

def compare_files(file1, file2):
    comparisons = []

    for channel_num in range(19):  # Assuming 19 channels
        seq1 = file1['channel_data'][channel_num]
        seq2 = file2['channel_data'][channel_num]

        # Ensure sequences are in string format
        if isinstance(seq1, np.ndarray):
            seq1 = ''.join(map(str, seq1))
        if isinstance(seq2, np.ndarray):
            seq2 = ''.join(map(str, seq2))

        percentage = needleman_wunsch(seq1, seq2)
        comparisons.append(f"Channel {channel_num + 1}: {percentage:.2f}%")

    # Join all comparisons with commas, but no leading comma
    comparisons_str = ", ".join(comparisons)
    return f"{file1['filename']} ({file1['group']}) - {file2['filename']} ({file2['group']}): {comparisons_str}"


def main():
    overall_start_time = time.time()  # Start the overall timer

    # Process all ADHD and Control files
    adhd_files = list(adhd_dir.glob("*.mat"))
    control_files = list(control_dir.glob("*.mat"))

    if len(adhd_files) == 0 or len(control_files) == 0:
        print("No files found in one or both directories.")
        return

    processed_files = []

    print("Processing ADHD files...")
    for adhd_file in adhd_files:
        adhd_key = process_single_file(adhd_file, adhd_output_dir, DOWNSAMPLE_RATE)
        if adhd_key:
            processed_files.append((adhd_key, 'ADHD', adhd_output_dir / f"{adhd_key}.txt"))

    print("Processing Control files...")
    for control_file in control_files:
        control_key = process_single_file(control_file, control_output_dir, DOWNSAMPLE_RATE)
        if control_key:
            processed_files.append((control_key, 'Control', control_output_dir / f"{control_key}.txt"))

    # If no files were successfully processed, exit
    if len(processed_files) == 0:
        print("File processing failed for all files.")
        return

    def read_file(filepath, group):
        with open(filepath, 'r') as f:
            channel_data = {}
            for line in f:
                parts = line.strip().split(': ', 1)
                if len(parts) == 2:
                    channel_data[int(parts[0].split(' ')[1]) - 1] = parts[1]

            return {
                'filename': filepath.stem,
                'group': group,
                'channel_data': [channel_data.get(i) for i in range(19)]  # Assuming 19 channels
            }

    file_data = [read_file(filepath, group) for key, group, filepath in processed_files]

    comparison_start_time = time.time()
    comparisons = []
    compared_pairs = set()  # Set to track compared pairs

    # Compare all pairs of files within the same group (ADHD-ADHD, Control-Control)
    for i in range(len(file_data)):
        for j in range(i + 1, len(file_data)):
            pair = (file_data[i]['filename'], file_data[j]['filename'])
            if pair not in compared_pairs:
                comparison_result = compare_files(file_data[i], file_data[j])
                comparisons.append(comparison_result)
                # Add both (file1, file2) and (file2, file1) to the set
                compared_pairs.add(pair)
                compared_pairs.add((file_data[j]['filename'], file_data[i]['filename']))

    # Compare all ADHD files with all Control files (ADHD-Control)
    for adhd_file_data in file_data:
        if adhd_file_data['group'] == 'ADHD':
            for control_file_data in file_data:
                if control_file_data['group'] == 'Control':
                    pair = (adhd_file_data['filename'], control_file_data['filename'])
                    if pair not in compared_pairs:
                        comparison_result = compare_files(adhd_file_data, control_file_data)
                        comparisons.append(comparison_result)
                        # Add both (file1, file2) and (file2, file1) to the set
                        compared_pairs.add(pair)
                        compared_pairs.add((control_file_data['filename'], adhd_file_data['filename']))

    # Write comparison results to file
    with open('Order_Preserving_Matching/comparison_results.txt', 'w') as result_file:
        result_file.write('\n'.join(comparisons) + '\n')

    print(f"Comparisons took {time.time() - comparison_start_time:.2f} seconds")

    overall_end_time = time.time()  # End the overall timer
    print(f"Overall processing took {overall_end_time - overall_start_time:.2f} seconds")

if __name__ == "__main__":
    main()
