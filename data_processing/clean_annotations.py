import csv, os
from collections import defaultdict

def process_csv(input_file_path, output_file_path):
    # Dictionary to hold the last occurrence of lines with the same 2nd and 3rd columns
    lines_dict = defaultdict(list)

    with open(input_file_path, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Assuming the CSV has a header

        for row in reader:
            key = (row[1], row[2])  # Tuple of 2nd and 3rd columns as key
            lines_dict[key] = row  # Only the last occurrence will be stored

    with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write header to the output file

        for row in lines_dict.values():
            # Modify the first column to keep only alphabetic characters
            filtering_predicate = lambda ch: ch.isalpha() or ch == " "
            row[0] = ''.join(filter(filtering_predicate, row[0]))
            writer.writerow(row)

# Example usage
if __name__ == "__main__":
    filename = 'Jasmi09.csv'
    input_file_path = os.path.join(os.getcwd(),"data/annotations" , filename)
    output_file_path = os.path.join(os.getcwd(),"data/annotations" , filename.replace(".csv","_clean.csv"))
    process_csv(input_file_path, output_file_path)
