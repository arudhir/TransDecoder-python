#!/usr/bin/env python3
"""
Simple example script that demonstrates how to use the Python TransDecoder API.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the path so we can import transdecoder
sys.path.append(str(Path(__file__).parent.parent))

from transdecoder.utils import read_fasta, write_fasta
from transdecoder.longorfs import LongOrfsExtractor
from transdecoder.markov import MarkovModel
from transdecoder.predict import TransDecoderPredictor


def main():
    """Run a simple TransDecoder example"""
    # Create a temporary directory for our example
    with tempfile.TemporaryDirectory() as temp_dir:
        # Sample transcript sequence with a known coding region
        transcript_seq = (
            "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
            "ATGCCAGAATTCGGCAAGGTTATGACCCGTCGTGCCACTCGTATGACGCGTCGTTACTAC"
            "CGTATCGATCGATCGATCGATCGATCGATCGATCGATCTCGATCGATCGACGATCGATCG"
            "TCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGTACGATCGA"
            "TCGATCGATCGATCGATCGATCGATCGATCGATCGATCGTAGCATAGATCGATCATGTAG"
            "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
        )
        
        # Create a FASTA file with our test sequence
        transcripts_file = os.path.join(temp_dir, "test_transcript.fasta")
        write_fasta({"transcript1": transcript_seq}, transcripts_file)
        
        print(f"Created test transcript file: {transcripts_file}")
        print(f"Transcript length: {len(transcript_seq)} bp")
        
        # Extract long ORFs
        print("\n1. Extracting long ORFs...")
        extractor = LongOrfsExtractor(
            min_protein_length=20,  # Lower threshold for this example
            genetic_code="universal",
            top_strand_only=False,
            complete_orfs_only=False
        )
        
        transcripts = read_fasta(transcripts_file)
        orfs_by_transcript = extractor.process_transcripts(transcripts)
        
        workdir = os.path.join(temp_dir, "transdecoder_dir")
        os.makedirs(workdir, exist_ok=True)
        
        gff3_file, cds_file, pep_file = extractor.write_output_files(
            orfs_by_transcript, workdir, "longest_orfs"
        )
        
        print(f"Found {sum(len(orfs) for orfs in orfs_by_transcript.values())} ORFs")
        print(f"GFF3 file: {gff3_file}")
        print(f"CDS file: {cds_file}")
        print(f"Protein file: {pep_file}")
        
        # Calculate base frequencies
        base_freqs = {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}  # Simple example
        base_freqs_file = os.path.join(workdir, "base_freqs.dat")
        with open(base_freqs_file, 'w') as f:
            for base, freq in base_freqs.items():
                f.write(f"{base}\t100\t{freq}\n")
        
        # Train Markov model
        print("\n2. Training Markov model...")
        cds_seqs = read_fasta(cds_file)
        
        model = MarkovModel()
        model.count_kmers(cds_seqs)
        model.set_background_probs(base_freqs)
        model.compute_loglikelihood_scores()
        
        hexamer_scores_file = os.path.join(workdir, "hexamer.scores")
        model.save_model(hexamer_scores_file)
        
        # Score ORFs
        print("\n3. Scoring ORFs...")
        scores_file = os.path.join(workdir, f"{os.path.basename(cds_file)}.scores")
        with open(scores_file, 'w') as f:
            # Write header
            f.write("#acc\tMarkov_order\tseq_length\tscore_1\tscore_2\tscore_3\tscore_4\tscore_5\tscore_6\n")
            
            for header, seq in cds_seqs.items():
                forward_scores, reverse_scores = model.score_sequence(seq)
                
                # Write scores
                f.write(f"{header}\t5\t{len(seq)}")
                for score in forward_scores + reverse_scores:
                    f.write(f"\t{score:.2f}")
                f.write("\n")
        
        # Run full TransDecoder.Predict
        print("\n4. Running TransDecoder.Predict...")
        predictor = TransDecoderPredictor(
            transcripts_file=transcripts_file,
            output_dir=temp_dir,
            retain_long_orfs_mode="strict",
            retain_long_orfs_length=60,  # Lower for this example
            top_orfs_train=1,  # Use all ORFs for training
            genetic_code="universal"
        )
        
        predictor.run_pipeline(
            no_refine_starts=True  # Skip start codon refinement for this example
        )
        
        print("\nExample completed successfully!")


if __name__ == "__main__":
    main()