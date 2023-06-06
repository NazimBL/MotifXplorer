import sys
import random
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


def generate_negative_examples(bed_file, reference_genome):
    reference = Fasta(reference_genome)

    neg_examples = []
    with open(bed_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#'):
                continue

            fields = line.split('\t')
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            length = end - start

            # Randomly select a new region that does not overlap with the input region
            while True:
                new_start = random.randint(0, len(reference[chrom]) - length)
                new_end = new_start + length
                if not any(new_start <= s < new_end or new_start < e <= new_end for (c, s, e) in neg_examples):
                    neg_examples.append((chrom, new_start, new_end))
                    break

    return neg_examples


def add_negative_examples(bed_lines, negative_examples, reference):
    output_lines = []
    for (line, (chrom, start, end)) in zip(bed_lines, negative_examples):
        line = line.strip()
        fields = line.split('\t')
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

    reference = Fasta(reference_genome)

    bed_lines = process_bed_file(bed_file, reference_genome)
    negative_examples = generate_negative_examples(bed_file, reference_genome)
    output_lines = add_negative_examples(bed_lines, negative_examples, reference)
    write_output(bed_file, output_lines)
