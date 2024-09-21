import os
import subprocess
from pathlib import Path
import scipy.io
import numpy as np
import scipy.signal as signal
from Bio.Blast import NCBIXML
import time
start_time = time.time()
# Function to map frequency to letters
def map_to_letter(frequency):
    mapping = {
        (0.5, 4): 'D',  # Delta
        (4, 8): 'T',  # Theta
        (8, 12): 'A',  # Alpha
        (12, 30): 'B',  # Beta
        (30, 40): 'G'  # Gamma
    }
    for freq_range, letter in mapping.items():
        if freq_range[0] <= frequency < freq_range[1]:
            return letter
    return None  # Return None for frequencies outside defined ranges

# Function to convert EEG data into sequence of letters
def eeg_to_letters(eeg_data, segment_size, overlap, sampling_freq):
    num_samples = eeg_data.shape[0]
    letters = []
    start_idx = 0
    step_size = int(segment_size * (1 - overlap))
    while start_idx + segment_size <= num_samples:
        segment = eeg_data[start_idx:start_idx + segment_size]
        f, psd = signal.welch(segment, fs=sampling_freq, nperseg=min(segment_size, len(segment)))
        dominant_frequency = f[np.argmax(psd)]
        letter = map_to_letter(dominant_frequency)
        if letter:  # Only append if letter is not None
            letters.append(letter)
        start_idx += step_size  # Move to next segment with overlap
    return letters

# Directories containing the .mat files
adhd_directory = "Clean_DB/Clean_ADHD_DB"
control_directory = "Clean_DB/Clean_Control_DB"

# Output directories for the letter sequences
adhd_output_text_directory = "BLAST_DB/EEG_Letter_Sequences_ADHD"
control_output_text_directory = "BLAST_DB/EEG_Letter_Sequences_Control"

# Create output directories if they do not exist
os.makedirs(adhd_output_text_directory, exist_ok=True)
os.makedirs(control_output_text_directory, exist_ok=True)

# Sampling frequency
sfreq = 128

# Function to process files in a given directory and save sequences
def process_files(directory, output_directory):
    dna_mapping = {
        'D': 'A',  # Delta waves map to Adenine
        'T': 'T',  # Theta waves map to Thymine
        'A': 'C',  # Alpha waves map to Cytosine
        'B': 'G',  # Beta waves map to Guanine
        'G': 'U'  # Gamma waves map to Uracil
    }

    for file_name in os.listdir(directory):
        if file_name.endswith('.mat'):
            file_path = os.path.join(directory, file_name)
            mat = scipy.io.loadmat(file_path)
            variable_name = [key for key in mat.keys() if not key.startswith('__')][0]  # Dynamically get the variable name
            data = mat[variable_name]
            # Process each channel separately and combine results
            combined_sequences = []

            for channel_idx in range(data.shape[0]):
                channel_data = data[channel_idx, :]

                segment_duration = 2  # Segment duration in seconds
                segment_size = segment_duration * sfreq  # Segment size in samples
                overlap = 0.5  # 50% overlap

                # Process the channel data
                letters = eeg_to_letters(channel_data, segment_size, overlap, sfreq)
                blast_letters = [dna_mapping.get(letter, 'N') for letter in letters if letter]
                sequence_str = ''.join(blast_letters)

                combined_sequences.append(f"Channel {channel_idx + 1}:\n{sequence_str}\n")

            # Save the combined sequences of letters for all channels to a single text file
            base_name = os.path.splitext(file_name)[0]
            output_text_file_path = os.path.join(output_directory, f"{base_name}.txt")
            with open(output_text_file_path, 'w') as f:
                f.write('\n'.join(combined_sequences))

            print(f"Combined sequences of letters for {file_name} saved to {output_text_file_path}")

# Measure time for processing files
process_files(adhd_directory, adhd_output_text_directory)
process_files(control_directory, control_output_text_directory)


# Define directories
adhd_dir = Path("C:/BLAST_DB/EEG_Letter_Sequences_ADHD")
control_dir = Path("C:/BLAST_DB/EEG_Letter_Sequences_Control")
blast_db_dir = Path("C:/BLAST_DB")

# Define a function to run BLAST and save results
def run_blast(query_path, db_path, db_output_path, output_path):
    # Rebuild the BLAST database
    makeblastdb_command = [
        "makeblastdb",
        "-in", str(db_path),
        "-dbtype", "nucl",
        "-out", str(db_output_path)
    ]

    result = subprocess.run(makeblastdb_command, capture_output=True, text=True)
    print(f"makeblastdb command output: {result.stdout}")
    print(f"makeblastdb command error (if any): {result.stderr}")

    if result.returncode != 0:
        print(f"Error making BLAST database: {result.stderr}")
        exit(1)
    else:
        print(f"BLAST database created successfully.")

    # Set the BLASTDB environment variable to the directory containing the database
    os.environ['BLASTDB'] = str(blast_db_dir)

    # Define the BLASTn command with more lenient parameters
    blastn_command = [
        "blastn",
        "-query", str(query_path),
        "-db", str(db_output_path),
        "-evalue", "100",  # More lenient e-value to capture more alignments
        "-word_size", "4",  # Smallest possible word size for increased sensitivity
        "-reward", "1",
        "-penalty", "-1",
        "-gapopen", "5",
        "-gapextend", "2",
        "-dust", "no",
        "-soft_masking", "false",
        "-perc_identity", "30",  # Lower identity percentage for more alignments
        "-qcov_hsp_perc", "30",  # Lower coverage percentage for more alignments
        "-max_target_seqs", "100",  # Increase the maximum number of aligned sequences to keep
        "-outfmt", "5",  # XML format for detailed output
        "-out", str(output_path)
    ]

    # Run the BLASTn command
    result = subprocess.run(blastn_command, capture_output=True, text=True)
    print(f"blastn command output: {result.stdout}")
    print(f"blastn command error (if any): {result.stderr}")

    # Check for errors
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        exit(1)
    else:
        print(f"BLAST completed successfully.")

# Define a function to parse and save the longest identity percentage results
def save_longest_identity_results(output_path, result_file_path, file_sources):
    results = {}

    with open(output_path) as result_handle:
        blast_records = NCBIXML.parse(result_handle)
        for blast_record in blast_records:
            query_name = blast_record.query.split()[0]  # Get the query name
            query_parts = query_name.split('_')
            query_base_name = query_parts[0]
            query_channel = int(query_parts[-1])

            for alignment in blast_record.alignments:
                subject_name = alignment.title.split()[1]  # Get the subject name without prefix
                subject_parts = subject_name.split('_')
                subject_base_name = subject_parts[0]
                subject_channel = int(subject_parts[-1])

                # Skip if the query and subject names are the same or if the channel numbers are not the same
                if query_base_name == subject_base_name or query_channel != subject_channel:
                    continue

                # Initialize the result entry if not already present
                if (query_base_name, subject_base_name) not in results:
                    results[(query_base_name, subject_base_name)] = [None] * 19

                # Find the longest alignment for the current channel
                longest_alignment = max(alignment.hsps, key=lambda hsp: hsp.align_length)
                identity_percentage = (longest_alignment.identities / longest_alignment.align_length) * 100
                if results[(query_base_name, subject_base_name)][query_channel - 1] is None or \
                        results[(query_base_name, subject_base_name)][query_channel - 1] < identity_percentage:
                    results[(query_base_name, subject_base_name)][query_channel - 1] = identity_percentage

    # Convert the results dictionary to a list and sort it
    sorted_results = sorted(results.items())

    # Write results to file in the specified order
    with open(result_file_path, 'w') as result_file:
        for (query_base_name, subject_base_name), channels in sorted_results:
            query_group = file_sources[query_base_name]
            subject_group = file_sources[subject_base_name]

            result_str = f"{query_base_name} ({query_group}) - {subject_base_name} ({subject_group}): " + ", ".join(
                [f"Channel {i + 1}: {channels[i]:.2f}%" if channels[i] is not None else f"Channel {i + 1}: 0.00%" for i in
                 range(19)]
            )

            result_file.write(result_str + "\n")

    print(f"Longest identity percentage results saved to {result_file_path}")

# Define a function to parse and save the highest identity percentage results
def save_highest_identity_results(output_path, result_file_path, file_sources):
    results = {}

    with open(output_path) as result_handle:
        blast_records = NCBIXML.parse(result_handle)
        for blast_record in blast_records:
            query_name = blast_record.query.split()[0]  # Get the query name
            query_parts = query_name.split('_')
            query_base_name = query_parts[0]
            query_channel = int(query_parts[-1])

            for alignment in blast_record.alignments:
                subject_name = alignment.title.split()[1]  # Get the subject name without prefix
                subject_parts = subject_name.split('_')
                subject_base_name = subject_parts[0]
                subject_channel = int(subject_parts[-1])

                # Skip if the query and subject names are the same or if the channel numbers are not the same
                if query_base_name == subject_base_name or query_channel != subject_channel:
                    continue

                # Initialize the result entry if not already present
                if (query_base_name, subject_base_name) not in results:
                    results[(query_base_name, subject_base_name)] = [None] * 19

                # Find the highest identity percentage for the current channel
                highest_identity = max((hsp.identities / hsp.align_length) * 100 for hsp in alignment.hsps)
                if results[(query_base_name, subject_base_name)][query_channel - 1] is None or \
                        results[(query_base_name, subject_base_name)][query_channel - 1] < highest_identity:
                    results[(query_base_name, subject_base_name)][query_channel - 1] = highest_identity

    # Convert the results dictionary to a list and sort it
    sorted_results = sorted(results.items())

    # Write results to file in the specified order
    with open(result_file_path, 'w') as result_file:
        for (query_base_name, subject_base_name), channels in sorted_results:
            query_group = file_sources[query_base_name]
            subject_group = file_sources[subject_base_name]

            result_str = f"{query_base_name} ({query_group}) - {subject_base_name} ({subject_group}): " + ", ".join(
                [f"Channel {i + 1}: {channels[i]:.2f}%" if channels[i] is not None else f"Channel {i + 1}: 0.00%" for i in
                 range(19)]
            )
            result_file.write(result_str + "\n")

    print(f"Highest identity percentage results saved to {result_file_path}")

# Define a function to concatenate files into query.fasta and mydb.fasta
def concatenate_files(adhd_dir, control_dir, query_path, db_path):
    file_sources = {}

    with open(query_path, 'w') as query_file, open(db_path, 'w') as db_file:
        for txt_file in adhd_dir.glob('*.txt'):
            with open(txt_file, 'r') as f:
                sequence_data = f.read().strip()
                lines = sequence_data.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('Channel'):
                        channel_number = line.split()[1][:-1]
                        sequence_name = f'>{txt_file.stem}_Channel_{channel_number}\n'
                        sequence_data = lines[i + 1]
                        query_file.write(sequence_name + sequence_data + '\n')
                        db_file.write(sequence_name + sequence_data + '\n')
                        file_sources[txt_file.stem] = 'ADHD'
        for txt_file in control_dir.glob('*.txt'):
            with open(txt_file, 'r') as f:
                sequence_data = f.read().strip()
                lines = sequence_data.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('Channel'):
                        channel_number = line.split()[1][:-1]
                        sequence_name = f'>{txt_file.stem}_Channel_{channel_number}\n'
                        sequence_data = lines[i + 1]
                        query_file.write(sequence_name + sequence_data + '\n')
                        db_file.write(sequence_name + sequence_data + '\n')
                        file_sources[txt_file.stem] = 'Control'

    return file_sources

# Define the paths
query_path = blast_db_dir / "query.fasta"
db_path = blast_db_dir / "mydb.fasta"
db_output_path = blast_db_dir / "mydb"
output_path = blast_db_dir / "results.xml"

# Concatenate files and run the BLAST workflow
file_sources = concatenate_files(adhd_dir, control_dir, query_path, db_path)
run_blast(query_path, db_path, db_output_path, output_path)

# Save the longest identity percentage results
result_file_path1 = blast_db_dir / "longest_alignment_2.txt"
save_longest_identity_results(output_path, result_file_path1, file_sources)

# Save the highest identity percentage results
result_file_path2 = blast_db_dir / "highest_identity_2.txt"
save_highest_identity_results(output_path, result_file_path2, file_sources)
end_time = time.time()
print(f"Time for process: {end_time - start_time:.2f} seconds")
