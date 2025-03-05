"""
Predict module - identifies likely coding regions among extracted ORFs.

This module implements the second phase of TransDecoder, which scores ORFs
using a Markov model and selects the most likely protein-coding regions.
"""

import os
import sys
import click
import gzip
import random
import shutil
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Iterator
from collections import defaultdict, Counter
from dataclasses import dataclass
import hashlib
import re

from .utils import (
    read_fasta, write_fasta, reverse_complement, translate_sequence,
    compute_gc_content, calculate_base_frequencies, ensure_directory, 
    parse_gene_transcript_map, GENETIC_CODES
)
from .markov import MarkovModel
from .pwm import PWM, train_start_pwm, analyze_pwm_performance
from .longorfs import ORF, LongOrfsExtractor


# GC content to minimum ORF length thresholds
# Based on the 0.999 quantile from random data
GC_TO_MIN_LONG_ORF_LENGTH = [
    (25, 465),
    (30, 510),
    (35, 555),
    (40, 590),
    (45, 645),
    (50, 749),
    (55, 797),
    (60, 927),
    (65, 1086),
    (70, 1358),
    (75, 1743),
    (80, 2422)
]


class TransDecoderPredictor:
    """
    Class for predicting coding regions from candidate ORFs.
    Equivalent to TransDecoder.Predict in the original implementation.
    """
    
    def __init__(self, transcripts_file: str, 
                 output_dir: str,
                 retain_long_orfs_mode: str = "dynamic",
                 retain_long_orfs_length: int = 1000000,
                 top_orfs_train: int = 500,
                 genetic_code: str = "universal"):
        """
        Initialize the predictor.
        
        Args:
            transcripts_file: Path to transcripts FASTA file
            output_dir: Path to output directory
            retain_long_orfs_mode: Mode for retaining long ORFs ('dynamic' or 'strict')
            retain_long_orfs_length: Minimum length to retain ORFs in 'strict' mode
            top_orfs_train: Number of top longest ORFs to train the Markov model
            genetic_code: Genetic code to use
        """
        self.transcripts_file = transcripts_file
        self.output_dir = output_dir
        self.retain_long_orfs_mode = retain_long_orfs_mode
        self.retain_long_orfs_length = retain_long_orfs_length
        self.top_orfs_train = top_orfs_train
        self.genetic_code = genetic_code
        
        # Set transdecoder directory
        trans_basename = os.path.basename(transcripts_file)
        if trans_basename.endswith('.gz'):
            trans_basename = trans_basename[:-3]
            
        self.workdir = os.path.join(output_dir, f"{trans_basename}.transdecoder_dir")
        
        # Check if transdecoder directory exists
        if not os.path.exists(self.workdir):
            raise ValueError(f"Directory {self.workdir} does not exist. Run TransDecoder.LongOrfs first.")
            
        # Input files from LongOrfs
        self.prefix = os.path.join(self.workdir, "longest_orfs")
        self.cds_file = f"{self.prefix}.cds"
        self.gff3_file = f"{self.prefix}.gff3"
        self.pep_file = f"{self.prefix}.pep"
        
        # Read transcripts
        self.transcripts = read_fasta(transcripts_file)
        
        # Final output prefix
        self.final_output_prefix = os.path.join(output_dir, f"{trans_basename}.transdecoder")
    
    def select_training_orfs(self, max_prot_length: int = 5000) -> str:
        """
        Select top longest ORFs for training the Markov model.
        
        Args:
            max_prot_length: Maximum protein length to consider for training
            
        Returns:
            Path to the file with selected training ORFs
        """
        # Read CDS sequences
        cds_seqs = read_fasta(self.cds_file)
        
        # Filter and sort by length
        filtered_cds = {}
        for header, seq in cds_seqs.items():
            if len(seq) <= max_prot_length * 3:  # Convert protein length to nucleotides
                filtered_cds[header] = seq
                
        # Sort by length (descending)
        sorted_cds = {k: filtered_cds[k] for k in sorted(filtered_cds, key=lambda x: len(filtered_cds[x]), reverse=True)}
        
        # Get 10x the number of entries for reducing redundancy
        red_num = self.top_orfs_train * 10
        red_cds = dict(list(sorted_cds.items())[:red_num])
        red_cds_file = f"{self.cds_file}.top_longest_{red_num}"
        write_fasta(red_cds, red_cds_file)
        
        # Remove redundancy by sequence similarity
        nr_cds = self._remove_redundant_orfs(red_cds)
        
        # Select top training ORFs
        top_cds = dict(list(nr_cds.items())[:self.top_orfs_train])
        top_cds_file = f"{self.cds_file}.top_{self.top_orfs_train}_longest"
        write_fasta(top_cds, top_cds_file)
        
        return top_cds_file
    
    def _remove_redundant_orfs(self, cds_seqs: Dict[str, str]) -> Dict[str, str]:
        """
        Remove redundant ORFs based on sequence similarity.
        This is a simple implementation that removes exact duplicates.
        """
        # In a full implementation, this would use CD-HIT or similar for clustering
        # Here we'll just remove exact duplicates as a simple approximation
        unique_seqs = {}
        seq_to_header = {}
        
        for header, seq in cds_seqs.items():
            if seq not in seq_to_header:
                unique_seqs[header] = seq
                seq_to_header[seq] = header
        
        return unique_seqs
    
    def train_markov_model(self, training_cds_file: str) -> Tuple[str, int]:
        """
        Train a Markov model for scoring ORFs.
        
        Args:
            training_cds_file: File with CDS sequences for training
            
        Returns:
            Tuple of (hexamer_scores_file, min_length_threshold)
        """
        # Read training sequences
        training_seqs = read_fasta(training_cds_file)
        
        # Get base frequencies
        base_freqs_file = os.path.join(self.workdir, "base_freqs.dat")
        base_freqs = {}
        with open(base_freqs_file, 'r') as f:
            for line in f:
                base, count, ratio = line.strip().split('\t')
                base_freqs[base] = float(ratio)
        
        # Determine min ORF length threshold
        if self.retain_long_orfs_mode == "dynamic":
            pct_gc = 2 * base_freqs['C'] * 100  # All Cs in both strands
            min_length = self._get_dynamic_orf_length(pct_gc)
        else:
            min_length = self.retain_long_orfs_length
            
        print(f"Using minimum ORF length threshold: {min_length}")
        
        # Initialize and train Markov model
        model = MarkovModel()
        model.count_kmers(training_seqs)
        model.set_background_probs(base_freqs)
        model.compute_loglikelihood_scores()
        
        # Save model
        hexamer_scores_file = os.path.join(self.workdir, "hexamer.scores")
        model.save_model(hexamer_scores_file)
        
        return hexamer_scores_file, min_length
    
    def _get_dynamic_orf_length(self, pct_gc: float) -> int:
        """Get the dynamic ORF length threshold based on GC content"""
        for gc, min_len in GC_TO_MIN_LONG_ORF_LENGTH:
            if pct_gc <= gc:
                return min_len
                
        # If GC content is very high, rely on Markov model entirely
        return 1000000  # effectively infinity
    
    def score_all_orfs(self, hexamer_scores_file: str) -> str:
        """
        Score all ORFs using the trained Markov model.
        
        Args:
            hexamer_scores_file: File with hexamer scores
            
        Returns:
            Path to the scores file
        """
        # Load model
        model = MarkovModel.load_model(hexamer_scores_file)
        
        # Read CDS sequences
        cds_seqs = read_fasta(self.cds_file)
        
        # Score all sequences
        scores_file = f"{self.cds_file}.scores"
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
                
        return scores_file
    
    def select_best_orfs(self, scores_file: str, min_length: int, 
                        pfam_hits_file: Optional[str] = None,
                        blastp_hits_file: Optional[str] = None,
                        single_best_only: bool = False) -> str:
        """
        Select the best ORFs based on scores and other criteria.
        
        Args:
            scores_file: File with ORF scores
            min_length: Minimum ORF length to accept automatically
            pfam_hits_file: Optional file with Pfam hits
            blastp_hits_file: Optional file with BlastP hits
            single_best_only: Whether to retain only the single best ORF per transcript
            
        Returns:
            Path to the GFF3 file with selected ORFs
        """
        # Parse scores
        scores_by_orf = {}
        with open(scores_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split('\t')
                acc = parts[0]
                scores = [float(s) for s in parts[3:9]]
                scores_by_orf[acc] = max(scores)  # Use maximum score across frames
        
        # Parse Pfam hits if provided
        pfam_hits = set()
        if pfam_hits_file:
            with open(pfam_hits_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        pfam_hits.add(parts[3])  # ORF accession in 4th column
        
        # Parse BlastP hits if provided
        blast_hits = set()
        if blastp_hits_file:
            with open(blastp_hits_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    blast_hits.add(parts[0])  # Query ID in 1st column
        
        # Read GFF3 to get ORF details
        orfs_by_transcript = defaultdict(list)
        current_gene = None
        gene_to_orf = {}
        
        with open(self.gff3_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                    
                parts = line.strip().split('\t')
                if len(parts) != 9:
                    continue
                    
                transcript_id, source, feature_type, start, end, score, strand, phase, attributes = parts
                
                if feature_type == 'gene':
                    attr_dict = self._parse_gff3_attributes(attributes)
                    current_gene = attr_dict.get('ID', '')
                
                elif feature_type == 'CDS' and current_gene:
                    attr_dict = self._parse_gff3_attributes(attributes)
                    orf_id = attr_dict.get('ID', '')
                    
                    if orf_id:
                        gene_to_orf[current_gene] = orf_id
                        
                        # Extract ORF attributes
                        orf_type = attr_dict.get('orf_type', '')
                        orf_length = int(attr_dict.get('orf_length', '0'))
                        
                        # Create ORF object
                        orf = {
                            'id': orf_id,
                            'gene_id': current_gene,
                            'transcript_id': transcript_id,
                            'start': int(start),
                            'end': int(end),
                            'strand': strand,
                            'type': orf_type,
                            'length': orf_length,
                            'score': scores_by_orf.get(orf_id, 0),
                            'has_pfam': orf_id in pfam_hits,
                            'has_blast': orf_id in blast_hits
                        }
                        
                        orfs_by_transcript[transcript_id].append(orf)
        
        # Select best ORFs
        selected_orf_ids = set()
        
        # Process each transcript
        for transcript_id, orfs in orfs_by_transcript.items():
            # First, filter by length and homology
            good_orfs = []
            
            for orf in orfs:
                # Accept if long enough
                if orf['length'] >= min_length:
                    good_orfs.append(orf)
                    continue
                
                # Accept if has homology evidence
                if orf['has_pfam'] or orf['has_blast']:
                    good_orfs.append(orf)
                    continue
                
                # Also accept if good score and complete
                if orf['score'] > 0 and orf['type'] == 'complete':
                    good_orfs.append(orf)
            
            # If single_best_only, pick the best one
            if single_best_only and good_orfs:
                # Sort by homology and score
                good_orfs.sort(key=lambda x: (
                    -(x['has_pfam'] + x['has_blast']),  # Homology first (negative to sort desc)
                    -x['length'],                       # Then length
                    -x['score']                         # Then score
                ))
                # Keep only the best one
                selected_orf_ids.add(good_orfs[0]['id'])
            else:
                # Keep all good ORFs
                for orf in good_orfs:
                    selected_orf_ids.add(orf['id'])
        
        # Write selected ORFs to new GFF3
        best_candidates_gff3 = f"{self.cds_file}.best_candidates.gff3"
        
        with open(self.gff3_file, 'r') as f_in, open(best_candidates_gff3, 'w') as f_out:
            # Write header
            f_out.write("##gff-version 3\n")
            
            # Track current gene
            current_gene = None
            selected_genes = set()
            
            for line in f_in:
                if line.startswith('#') or not line.strip():
                    continue
                    
                parts = line.strip().split('\t')
                if len(parts) != 9:
                    continue
                    
                feature_type = parts[2]
                attributes = parts[8]
                
                if feature_type == 'gene':
                    attr_dict = self._parse_gff3_attributes(attributes)
                    current_gene = attr_dict.get('ID', '')
                    
                    # Check if this gene has a selected ORF
                    if current_gene in gene_to_orf and gene_to_orf[current_gene] in selected_orf_ids:
                        selected_genes.add(current_gene)
                        f_out.write(line)
                
                elif feature_type == 'CDS':
                    attr_dict = self._parse_gff3_attributes(attributes)
                    orf_id = attr_dict.get('ID', '')
                    
                    # Write only selected ORFs
                    if orf_id in selected_orf_ids:
                        f_out.write(line)
        
        return best_candidates_gff3
    
    def refine_start_codons(self, gff3_file: str) -> str:
        """
        Refine start codon selection using a position weight matrix.
        
        Args:
            gff3_file: GFF3 file with selected ORFs
            
        Returns:
            Path to the GFF3 file with refined start codons
        """
        # Get training sequences for start codon PWM
        top_cds_file = f"{self.cds_file}.top_{self.top_orfs_train}_longest"
        training_seqs = read_fasta(top_cds_file)
        
        # Extract sequences around start codons
        atg_context_size = 20  # nucleotides on each side
        positive_seqs = []
        
        for header, seq in training_seqs.items():
            # Find ATG position (should be at the start)
            start_pos = seq.upper().find('ATG')
            if start_pos >= 0:
                # Extract sequence around ATG with context
                extract_start = max(0, start_pos - atg_context_size)
                extract_end = min(len(seq), start_pos + 3 + atg_context_size)
                
                if extract_end - extract_start >= 25:  # Ensure enough context
                    context_seq = seq[extract_start:extract_end]
                    # Pad if needed
                    if extract_start > 0:
                        context_seq = 'N' * (atg_context_size - start_pos) + context_seq
                    if extract_end < len(seq):
                        context_seq = context_seq + 'N' * (start_pos + 3 + atg_context_size - extract_end)
                        
                    positive_seqs.append(context_seq)
        
        # Create negative sequences (non-start ATGs)
        negative_seqs = []
        
        for header, seq in training_seqs.items():
            seq = seq.upper()
            # Skip the first ATG (actual start)
            start_pos = seq.find('ATG')
            if start_pos >= 0:
                pos = start_pos + 3
                while True:
                    pos = seq.find('ATG', pos)
                    if pos == -1:
                        break
                        
                    # Extract context
                    extract_start = max(0, pos - atg_context_size)
                    extract_end = min(len(seq), pos + 3 + atg_context_size)
                    
                    if extract_end - extract_start >= 25:
                        context_seq = seq[extract_start:extract_end]
                        # Pad if needed
                        if extract_start > 0:
                            context_seq = 'N' * (atg_context_size - (pos - extract_start)) + context_seq
                        if extract_end < len(seq):
                            context_seq = context_seq + 'N' * (pos + 3 + atg_context_size - extract_end)
                            
                        negative_seqs.append(context_seq)
                    
                    pos += 3
        
        # Train PWMs
        if positive_seqs and negative_seqs:
            print(f"Training start codon PWMs with {len(positive_seqs)} positive and {len(negative_seqs)} negative examples")
            pos_pwm, neg_pwm, scores_data = train_start_pwm(
                positive_seqs, negative_seqs, 
                num_rounds=5, 
                fraction_train=0.75,
                atg_position=atg_context_size
            )
            
            # Save PWMs
            pwm_dir = os.path.join(self.workdir, "start_refinement")
            ensure_directory(pwm_dir)
            
            pos_pwm.save_pwm(os.path.join(pwm_dir, "start_pwm.+.pwm"))
            neg_pwm.save_pwm(os.path.join(pwm_dir, "start_pwm.-.pwm"))
            
            # Get best PWM range
            best_range, best_auc = analyze_pwm_performance(scores_data)
            
            if best_range:
                print(f"Best PWM range: {best_range}, AUC: {best_auc:.4f}")
                with open(os.path.join(pwm_dir, "pwm_range.txt"), 'w') as f:
                    f.write(f"{best_range}\t{best_auc}\n")
            
            # Refine start codons in selected transcripts
            refined_gff3 = self._apply_start_codon_pwm(gff3_file, pos_pwm, neg_pwm, pwm_dir)
            return refined_gff3
        else:
            print("Warning: Not enough sequences to train start codon PWMs")
            return gff3_file
    
    def _apply_start_codon_pwm(self, gff3_file: str, pos_pwm: PWM, neg_pwm: PWM, pwm_dir: str) -> str:
        """Apply the PWM to refine start codons in selected ORFs"""
        # Get the best PWM range
        pwm_range_file = os.path.join(pwm_dir, "pwm_range.txt")
        best_range = None
        with open(pwm_range_file, 'r') as f:
            line = f.readline().strip()
            if line:
                best_range = line.split('\t')[0]
        
        if not best_range:
            return gff3_file
            
        up, down = map(int, best_range.split(','))
        atg_context_size = 20  # Should match training
        
        # Parse GFF3 and group by transcript
        transcript_to_orfs = defaultdict(list)
        orf_data = {}
        
        with open(gff3_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                    
                parts = line.strip().split('\t')
                if len(parts) != 9:
                    continue
                    
                transcript_id, source, feature_type, start, end, score, strand, phase, attributes = parts
                
                if feature_type == 'CDS':
                    attr_dict = self._parse_gff3_attributes(attributes)
                    orf_id = attr_dict.get('ID', '')
                    
                    if orf_id:
                        orf = {
                            'id': orf_id,
                            'transcript_id': transcript_id,
                            'start': int(start),
                            'end': int(end),
                            'strand': strand,
                            'attributes': attributes,
                            'line': line
                        }
                        transcript_to_orfs[transcript_id].append(orf)
                        orf_data[orf_id] = orf
        
        # Refined ORFs output
        revised_starts_gff3 = f"{gff3_file}.revised_starts.gff3"
        
        # Copy original file
        shutil.copy(gff3_file, revised_starts_gff3)
        
        # For a more complete implementation, we would:
        # 1. Extract transcript sequences
        # 2. For each ORF, scan upstream for alternative start codons
        # 3. Score each potential start using the PWM
        # 4. Select the best scoring start codon
        # 5. Update the GFF3 file with revised start positions
        
        # This would require more complex GFF3 handling and sequence analysis
        # than we can implement in this example
        
        return revised_starts_gff3
    
    def create_output_files(self, gff3_file: str) -> None:
        """
        Create final output files (GFF3, BED, protein, CDS).
        
        Args:
            gff3_file: GFF3 file with selected ORFs
        """
        # Copy GFF3 to final location
        final_gff3 = f"{self.final_output_prefix}.gff3"
        shutil.copy(gff3_file, final_gff3)
        
        # Create BED file
        final_bed = f"{self.final_output_prefix}.bed"
        self._gff3_to_bed(final_gff3, final_bed)
        
        # Create protein file
        final_pep = f"{self.final_output_prefix}.pep"
        self._gff3_to_proteins(final_gff3, self.transcripts_file, final_pep, 'pep')
        
        # Create CDS file
        final_cds = f"{self.final_output_prefix}.cds"
        self._gff3_to_proteins(final_gff3, self.transcripts_file, final_cds, 'cds')
        
        print(f"\nTransDecoder output files:")
        print(f"  GFF3: {final_gff3}")
        print(f"  BED: {final_bed}")
        print(f"  Protein: {final_pep}")
        print(f"  CDS: {final_cds}")
    
    def _gff3_to_bed(self, gff3_file: str, bed_file: str) -> None:
        """Convert GFF3 to BED format"""
        with open(gff3_file, 'r') as f_in, open(bed_file, 'w') as f_out:
            for line in f_in:
                if line.startswith('#') or not line.strip():
                    continue
                    
                parts = line.strip().split('\t')
                if len(parts) != 9 or parts[2] != 'CDS':
                    continue
                    
                chrom = parts[0]
                start = int(parts[3]) - 1  # BED is 0-based
                end = int(parts[4])
                attr_dict = self._parse_gff3_attributes(parts[8])
                name = attr_dict.get('ID', 'unknown')
                score = parts[5] if parts[5] != '.' else '0'
                strand = parts[6]
                
                f_out.write(f"{chrom}\t{start}\t{end}\t{name}\t{score}\t{strand}\n")
    
    def _gff3_to_proteins(self, gff3_file: str, transcripts_file: str, output_file: str, seq_type: str) -> None:
        """Extract protein or CDS sequences based on GFF3 annotations"""
        # Read transcripts
        transcripts = read_fasta(transcripts_file)
        
        # Parse GFF3 and extract features
        features = []
        current_gene = None
        
        with open(gff3_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                    
                parts = line.strip().split('\t')
                if len(parts) != 9:
                    continue
                    
                transcript_id, source, feature_type, start, end, score, strand, phase, attributes = parts
                
                if feature_type == 'gene':
                    attr_dict = self._parse_gff3_attributes(attributes)
                    current_gene = attr_dict.get('ID', '')
                
                elif feature_type == 'CDS' and current_gene:
                    attr_dict = self._parse_gff3_attributes(attributes)
                    orf_id = attr_dict.get('ID', '')
                    
                    if orf_id and transcript_id in transcripts:
                        feature = {
                            'gene_id': current_gene,
                            'id': orf_id,
                            'transcript_id': transcript_id,
                            'start': int(start),
                            'end': int(end),
                            'strand': strand,
                            'attributes': attr_dict
                        }
                        features.append(feature)
        
        # Extract sequences
        sequences = {}
        
        for feature in features:
            transcript_id = feature['transcript_id']
            transcript_seq = transcripts.get(transcript_id, '')
            
            if not transcript_seq:
                continue
                
            # Extract CDS sequence
            start = feature['start'] - 1  # Convert to 0-based
            end = feature['end']
            cds_seq = transcript_seq[start:end]
            
            if feature['strand'] == '-':
                cds_seq = reverse_complement(cds_seq)
            
            # For protein sequences, translate
            if seq_type == 'pep':
                seq = translate_sequence(cds_seq, self.genetic_code)
            else:
                seq = cds_seq
                
            orf_id = feature['id']
            sequences[orf_id] = seq
            
        # Write output
        write_fasta(sequences, output_file)
    
    def _parse_gff3_attributes(self, attr_str: str) -> Dict[str, str]:
        """Parse GFF3 attribute string into a dictionary"""
        result = {}
        for pair in attr_str.split(';'):
            if not pair.strip():
                continue
            if '=' in pair:
                key, value = pair.split('=', 1)
                result[key] = value
        return result
    
    def run_pipeline(self, pfam_hits_file: Optional[str] = None,
                   blastp_hits_file: Optional[str] = None,
                   single_best_only: bool = False,
                   no_refine_starts: bool = False) -> None:
        """
        Run the full TransDecoder.Predict pipeline.
        
        Args:
            pfam_hits_file: Optional file with Pfam hits
            blastp_hits_file: Optional file with BlastP hits
            single_best_only: Whether to retain only the single best ORF per transcript
            no_refine_starts: Whether to skip start codon refinement
        """
        # 1. Select training ORFs
        print("Selecting training ORFs...")
        training_cds_file = self.select_training_orfs()
        
        # 2. Train Markov model
        print("Training Markov model...")
        hexamer_scores_file, min_length = self.train_markov_model(training_cds_file)
        
        # 3. Score all ORFs
        print("Scoring all ORFs...")
        scores_file = self.score_all_orfs(hexamer_scores_file)
        
        # 4. Select best ORFs
        print("Selecting best ORFs...")
        best_orfs_gff3 = self.select_best_orfs(
            scores_file, min_length,
            pfam_hits_file, blastp_hits_file,
            single_best_only
        )
        
        # 5. Refine start codons (optional)
        if not no_refine_starts:
            print("Refining start codons...")
            final_gff3 = self.refine_start_codons(best_orfs_gff3)
        else:
            final_gff3 = best_orfs_gff3
        
        # 6. Create output files
        print("Creating output files...")
        self.create_output_files(final_gff3)
        
        print("\nTransDecoder.Predict completed successfully.")


@click.command()
@click.option('-t', '--transcripts', required=True, help='Transcripts fasta file')
@click.option('--retain_long_orfs_mode', default='dynamic', 
             type=click.Choice(['dynamic', 'strict']),
             help='Mode for retaining long ORFs')
@click.option('--retain_long_orfs_length', default=1000000, 
             help='Minimum ORF length to retain in strict mode')
@click.option('--retain_pfam_hits', help='Pfam domain table hits file')
@click.option('--retain_blastp_hits', help='BlastP hits file')
@click.option('--single_best_only', is_flag=True, 
             help='Retain only single best ORF per transcript')
@click.option('--genetic_code', '-G', default='universal', help='Genetic code to use')
@click.option('--output_dir', '-O', help='Output directory')
@click.option('-T', '--top_orfs_train', default=500, 
             help='Top longest ORFs to train Markov model')
@click.option('--no_refine_starts', is_flag=True, 
             help='Skip start codon refinement')
def main(transcripts, retain_long_orfs_mode, retain_long_orfs_length,
        retain_pfam_hits, retain_blastp_hits, single_best_only,
        genetic_code, output_dir, top_orfs_train, no_refine_starts):
    """Predict likely coding regions from transcript sequences"""
    
    # Validate genetic code
    if genetic_code not in GENETIC_CODES:
        valid_codes = ", ".join(GENETIC_CODES.keys())
        click.echo(f"Error: Invalid genetic code. Valid options are: {valid_codes}")
        sys.exit(1)
    
    # Set output directory
    if not output_dir:
        output_dir = os.path.dirname(os.path.abspath(transcripts))
    
    # Initialize predictor
    predictor = TransDecoderPredictor(
        transcripts_file=transcripts,
        output_dir=output_dir,
        retain_long_orfs_mode=retain_long_orfs_mode,
        retain_long_orfs_length=retain_long_orfs_length,
        top_orfs_train=top_orfs_train,
        genetic_code=genetic_code
    )
    
    # Run pipeline
    predictor.run_pipeline(
        pfam_hits_file=retain_pfam_hits,
        blastp_hits_file=retain_blastp_hits,
        single_best_only=single_best_only,
        no_refine_starts=no_refine_starts
    )


if __name__ == "__main__":
    main()