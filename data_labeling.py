import sys
import csv

def create_csv(bed_file, output_csv):
    sequences = []
    labels = []

    with open(bed_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                continue

            fields = line.split('\t')
            positive_sequence = fields[4]
            negative_sequence = fields[5]

            if positive_sequence:
                sequences.append(positive_sequence)
                labels.append('1')

            if negative_sequence:
                sequences.append(negative_sequence)
                labels.append('0')

    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['sequence', 'label'])
        writer.writerows(zip(sequences, labels))

    print(f"CSV file created: {output_csv}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python data_labeling.py <bed_file> <output_csv>")
        sys.exit(1)

    bed_file = sys.argv[1]
    output_csv = sys.argv[2]

    create_csv(bed_file, output_csv)
