import sys
from pyfaidx import Fasta


def get_sequence(chrom, start, end, reference):
    sequence = reference[chrom][start:end].seq
    return sequence


def process_bed_file(bed_file, reference_genome):
    reference = Fasta(reference_genome)

    output_lines = []
    with open(bed_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                output_lines.append(line)
                continue

            fields = line.split('\t')
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])

            sequence = get_sequence(chrom, start, end, reference)
            fields.append(sequence)
            output_lines.append('\t'.join(fields))

    return output_lines


def write_output(bed_file, output_lines):
    output_file = bed_file + '.new'
    with open(output_file, 'w') as file:
        for line in output_lines:
            file.write(line + '\n')
    print(f"Output written to {output_file}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python extract_sequences.py <bed_file> <reference_genome>")
        sys.exit(1)

    bed_file = sys.argv[1]
    reference_genome = sys.argv[2]

    output_lines = process_bed_file(bed_file, reference_genome)
    write_output(bed_file, output_lines)
