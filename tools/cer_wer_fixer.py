import csv

model = 'mary_tts'
# Define the input and output file paths
# input_file = f'../inference/{model}/cer_wer.csv'
# output_file = f'../inference/{model}/cer_wer_fix.csv'
input_file = '../inference/_data/cer_wer.csv'
output_file = '../inference/_data/cer_wer_fix.csv'

# Open the input file for reading and the output file for writing
with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
        open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

    # Create a CSV reader and writer
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    # Write the header for the new CSV
    writer.writerow(['deg', 'cer', 'wer'])

    # Iterate through each row of the input CSV
    for row in reader:
        if row:  # Check if the row is not empty
            # Extract the filename, cer, and wer values
            filename = row[1].split('/')[-1]  # Get the filename from the path
            cer = row[4]
            wer = row[5]

            # Write the extracted values to the output CSV
            writer.writerow([filename, cer, wer])

print(f"Conversion complete. Data saved to {output_file}")
