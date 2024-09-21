import re
from collections import defaultdict
from pathlib import Path
# Path to the blast_results.txt file
#results_path = 'C:/BLAST_DB/highest_identity_1.txt'

# Path to the Order_Preserving_Matching result
results_path =  Path("Order_Preserving_Matching/comparison_results.txt")

# Regular expression to parse each line of the blast_results.txt file
line_regex = re.compile(r'(\S+) \((\S+)\) - (\S+) \((\S+)\): (.*)')

# Dictionaries to store sums and counts for ADHD and Control and to store the type of each query name
sum_adhd = defaultdict(lambda: [0] * 19)
count_adhd = defaultdict(lambda: [0] * 19)
sum_control = defaultdict(lambda: [0] * 19)
count_control = defaultdict(lambda: [0] * 19)
query_types = {}

# Counters for total number of ADHD and Control files
total_adhd_files = set()
total_control_files = set()

# Process the blast_results.txt file
with open(results_path, 'r') as file:
    for line in file:
        match = line_regex.match(line)
        if match:
            query_name, query_group, subject_name, subject_group, channels_info = match.groups()
            channels = channels_info.split(', ')
            query_types[query_name] = query_group  # Store the type of the query name
            for channel_info in channels:
                channel_match = re.match(r'Channel (\d+): ([\d.]+)%', channel_info)
                if channel_match:
                    channel = int(channel_match.group(1)) - 1  # Convert channel to 0-based index
                    identity = float(channel_match.group(2))

                    if subject_group == 'ADHD':
                        sum_adhd[query_name][channel] += identity
                        count_adhd[query_name][channel] += 1
                    if subject_group == 'Control':
                        sum_control[query_name][channel] += identity
                        count_control[query_name][channel] += 1

            # Increment the total count based on the group
            if query_group == 'ADHD':
                total_adhd_files.add(query_name)
            elif query_group == 'Control':
                total_control_files.add(query_name)

# Counter for the specific condition for channel
ADHD_count = [0] * 19
Control_count = [0] * 19

# Print the results in the desired format
for query_name in sorted(set(sum_adhd.keys()).union(sum_control.keys())):
    adhd_sums = sum_adhd[query_name]
    adhd_counts = count_adhd[query_name]
    control_sums = sum_control[query_name]
    control_counts = count_control[query_name]
    query_group = query_types[query_name]

    channel_sums = []
    for channel in range(19):
        adhd_avg = adhd_sums[channel] / adhd_counts[channel] if adhd_counts[channel] > 0 else 0
        control_avg = control_sums[channel] / control_counts[channel] if control_counts[channel] > 0 else 0
        if adhd_avg > control_avg:
            channel_sums.append(f"Channel {channel + 1}: ADHD {adhd_avg:.2f}%")
            if query_group == 'ADHD':  # Channel 16 is at index 15 (0-based)
                ADHD_count [channel] += 1
        else:
            channel_sums.append(f"Channel {channel + 1}: Control {control_avg:.2f}%")
            if query_group == 'Control':  # Channel 13 is at index 12 (0-based)
                Control_count[channel] += 1

    print(f"{query_name} ({query_group}) - " + ", ".join(channel_sums))

# Print the count for correct classifications at the end
for i in range(19):
    print(f"With channel {i + 1} classified ADHD:{ADHD_count[i]}")
    print(f"With channel {i + 1} classified Control:{Control_count[i]}")

total_adhd = len(total_adhd_files)
total_control = len(total_control_files)
print(f"***Best results***")
print(f"Correctly classified ADHD: {max(ADHD_count)} out of {total_adhd} by channel number {ADHD_count.index(max(ADHD_count))+1}")
print(f"Correctly classified Control: {max(Control_count)} out of {total_control} by channel number {Control_count.index((max(Control_count)))+1}" )
print(f"In total {max(ADHD_count) + max(Control_count)} out of {total_adhd + total_control} diagnoses were correct")
