import os
import scipy.io
import mne


# Function to clean EEG data
def clean_eeg_data(file_path, output_directory, n_components=19):
    mat = scipy.io.loadmat(file_path)
    variable_name = [key for key in mat.keys() if not key.startswith('__')][0]
    data = mat[variable_name]
    sfreq = 128

    # Specified channel order
    ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
                'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz']

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data.T, info)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    # Filter the data
    raw.filter(0.5, 40., fir_design='firwin')

    # Apply ICA to remove artifacts
    ica = mne.preprocessing.ICA(n_components=n_components, random_state=97, max_iter=800)
    ica.fit(raw)

    # Manually select components to exclude (this needs to be adjusted based on visual inspection)
    ica.exclude = [0, 1]  # Example component indices to exclude

    # Apply ICA to remove artifacts
    raw_cleaned = ica.apply(raw.copy())

    cleaned_data = raw_cleaned.get_data()

    base_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_directory, base_name)
    scipy.io.savemat(output_file_path, {variable_name: cleaned_data})


adhd_directory = "DB/ADHD_DB"
control_directory = "DB/Control_DB"
cleaned_adhd_directory = "Clean_DB/Clean_ADHD_DB"
cleaned_control_directory = "Clean_DB/Clean_Control_DB"

if not os.path.exists(cleaned_adhd_directory):
    os.makedirs(cleaned_adhd_directory)
if not os.path.exists(cleaned_control_directory):
    os.makedirs(cleaned_control_directory)

for file_name in os.listdir(adhd_directory):
    if file_name.endswith('.mat'):
        file_path = os.path.join(adhd_directory, file_name)
        clean_eeg_data(file_path, cleaned_adhd_directory, n_components=19)
        print(f"Processed and saved cleaned data for {file_name} in {cleaned_adhd_directory}")

for file_name in os.listdir(control_directory):
    if file_name.endswith('.mat'):
        file_path = os.path.join(control_directory, file_name)
        clean_eeg_data(file_path, cleaned_control_directory, n_components=19)
        print(f"Processed and saved cleaned data for {file_name} in {cleaned_control_directory}")

print("All files have been processed and saved.")
