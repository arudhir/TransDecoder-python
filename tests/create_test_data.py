#!/usr/bin/env python3
"""
Create synthetic test data for benchmarking TransDecoder implementations.
"""

import os
import sys
import random
import argparse
from pathlib import Path
import gzip
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from transdecoder.utils import write_fasta


def generate_random_dna(length: int, gc_content: float = 0.5) -> str:
    """Generate random DNA sequence with specified length and GC content"""
    # Probabilities for A, C, G, T
    p_gc = gc_content / 2.0
    p_at = (1.0 - gc_content) / 2.0
    probs = {'A': p_at, 'C': p_gc, 'G': p_gc, 'T': p_at}
    
    # Generate random sequence
    bases = list(probs.keys())
    weights = [probs[base] for base in bases]
    
    return ''.join(random.choices(bases, weights=weights, k=length))


def generate_coding_sequence(length: int, gc_content: float = 0.5) -> str:
    """Generate a random coding sequence with start and stop codons"""
    # Ensure length is multiple of 3
    length = (length // 3) * 3
    
    if length < 6:
        raise ValueError("Length must be at least 6 (start + stop codons)")
    
    # Start with ATG
    sequence = "ATG"
    
    # Generate random codons (avoiding stop codons)
    stop_codons = ["TAA", "TAG", "TGA"]
    possible_codons = []
    
    # Generate all possible codons
    for a in "ACGT":
        for b in "ACGT":
            for c in "ACGT":
                codon = a + b + c
                if codon not in stop_codons and codon != "ATG":
                    possible_codons.append(codon)
    
    # Adjust probabilities based on GC content
    codon_weights = []
    for codon in possible_codons:
        gc_count = codon.count('G') + codon.count('C')
        weight = (gc_content ** gc_count) * ((1 - gc_content) ** (3 - gc_count))
        codon_weights.append(weight)
    
    # Normalize weights
    total_weight = sum(codon_weights)
    codon_weights = [w / total_weight for w in codon_weights]
    
    # Generate the middle part
    middle_length = length - 6  # Subtract ATG and stop codon
    codons_needed = middle_length // 3
    
    middle_sequence = ''.join(random.choices(possible_codons, weights=codon_weights, k=codons_needed))
    sequence += middle_sequence
    
    # End with a stop codon
    sequence += random.choice(stop_codons)
    
    return sequence


def generate_transcript_with_orf(
    orf_length: int, 
    utr5_length: int = 100, 
    utr3_length: int = 200,
    gc_content: float = 0.5
) -> str:
    """Generate a transcript containing a coding ORF with UTRs"""
    # Generate UTRs
    utr5 = generate_random_dna(utr5_length, gc_content)
    utr3 = generate_random_dna(utr3_length, gc_content)
    
    # Generate coding sequence
    coding_seq = generate_coding_sequence(orf_length, gc_content)
    
    # Combine all parts
    transcript = utr5 + coding_seq + utr3
    
    return transcript


def generate_transcript_set(
    num_transcripts: int,
    min_orf_length: int = 300,
    max_orf_length: int = 1500,
    min_gc: float = 0.4,
    max_gc: float = 0.6,
    output_file: str = "test_transcripts.fa"
) -> None:
    """Generate a set of transcripts with coding ORFs of varying lengths"""
    transcripts = {}
    
    for i in range(num_transcripts):
        # Random parameters for this transcript
        orf_length = random.randint(min_orf_length, max_orf_length)
        gc_content = random.uniform(min_gc, max_gc)
        utr5_length = random.randint(50, 300)
        utr3_length = random.randint(100, 500)
        
        # Generate transcript
        transcript = generate_transcript_with_orf(
            orf_length=orf_length,
            utr5_length=utr5_length,
            utr3_length=utr3_length,
            gc_content=gc_content
        )
        
        # Add to dictionary
        transcript_id = f"transcript_{i+1}_len_{len(transcript)}_orf_{orf_length}_gc_{gc_content:.2f}"
        transcripts[transcript_id] = transcript
    
    # Write to file
    write_fasta(transcripts, output_file)
    
    print(f"Generated {num_transcripts} transcripts in {output_file}")
    print(f"Total base pairs: {sum(len(seq) for seq in transcripts.values())}")


def generate_benchmark_sets(output_dir: str) -> None:
    """Generate benchmark datasets of different sizes"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Small set (for quick tests)
    generate_transcript_set(
        num_transcripts=100,
        min_orf_length=300,
        max_orf_length=1500,
        output_file=os.path.join(output_dir, "transcripts_small.fa")
    )
    
    # Medium set
    generate_transcript_set(
        num_transcripts=1000,
        min_orf_length=300,
        max_orf_length=2000,
        output_file=os.path.join(output_dir, "transcripts_medium.fa")
    )
    
    # Large set
    generate_transcript_set(
        num_transcripts=10000,
        min_orf_length=300,
        max_orf_length=3000,
        output_file=os.path.join(output_dir, "transcripts_large.fa")
    )
    
    # Mixed GC content set
    generate_transcript_set(
        num_transcripts=1000,
        min_orf_length=300,
        max_orf_length=2000,
        min_gc=0.25,
        max_gc=0.75,
        output_file=os.path.join(output_dir, "transcripts_mixed_gc.fa")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate test data for TransDecoder benchmarking")
    parser.add_argument("--output_dir", default="test_data/input", help="Output directory for test files")
    parser.add_argument("--num_transcripts", type=int, default=1000, help="Number of transcripts to generate")
    parser.add_argument("--min_orf", type=int, default=300, help="Minimum ORF length")
    parser.add_argument("--max_orf", type=int, default=2000, help="Maximum ORF length")
    parser.add_argument("--create_benchmarks", action="store_true", help="Create benchmark datasets")
    
    args = parser.parse_args()
    
    if args.create_benchmarks:
        generate_benchmark_sets(args.output_dir)
    else:
        generate_transcript_set(
            num_transcripts=args.num_transcripts,
            min_orf_length=args.min_orf,
            max_orf_length=args.max_orf,
            output_file=os.path.join(args.output_dir, "transcripts.fa")
        )