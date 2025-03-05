#!/usr/bin/env python3
"""
Benchmark script for comparing Perl and Python TransDecoder implementations.
"""

import os
import sys
import time
import argparse
import subprocess
import tempfile
import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import resource
import psutil
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from transdecoder.utils import read_fasta
from transdecoder.longorfs import LongOrfsExtractor
from transdecoder.predict import TransDecoderPredictor


def run_command(cmd: str, cwd: str = None) -> Tuple[str, str, int, float]:
    """Run a command and return stdout, stderr, return code, and execution time"""
    start_time = time.time()
    process = subprocess.Popen(
        cmd, shell=True, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True
    )
    stdout, stderr = process.communicate()
    end_time = time.time()
    
    return stdout, stderr, process.returncode, end_time - start_time


def get_memory_usage() -> int:
    """Get current memory usage in bytes"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


def benchmark_perl_longorfs(
    transcripts_file: str, 
    output_dir: str,
    min_protein_length: int = 100,
    complete_orfs_only: bool = False,
    top_strand_only: bool = False
) -> Dict[str, Any]:
    """Benchmark the Perl TransDecoder.LongOrfs implementation"""
    results = {
        "implementation": "perl",
        "phase": "longorfs",
        "start_time": time.time(),
        "input_file": transcripts_file,
        "input_size": os.path.getsize(transcripts_file),
        "num_sequences": len(read_fasta(transcripts_file)),
    }
    
    # Build command
    perl_bin = Path(__file__).parent.parent.parent / "TransDecoder.LongOrfs"
    cmd = f"{perl_bin} -t {transcripts_file} --output_dir {output_dir} -m {min_protein_length}"
    
    if complete_orfs_only:
        cmd += " --complete_orfs_only"
    if top_strand_only:
        cmd += " -S"
    
    # Run command
    memory_before = get_memory_usage()
    stdout, stderr, retcode, execution_time = run_command(cmd)
    memory_after = get_memory_usage()
    
    # Collect results
    results.update({
        "execution_time": execution_time,
        "return_code": retcode,
        "memory_usage": memory_after - memory_before,
        "end_time": time.time(),
        "command": cmd,
        "success": retcode == 0
    })
    
    # Get output file sizes
    td_dir = os.path.join(output_dir, os.path.basename(transcripts_file) + ".transdecoder_dir")
    
    if os.path.exists(td_dir):
        prefix = os.path.join(td_dir, "longest_orfs")
        
        results.update({
            "output_gff3_size": os.path.getsize(f"{prefix}.gff3") if os.path.exists(f"{prefix}.gff3") else 0,
            "output_cds_size": os.path.getsize(f"{prefix}.cds") if os.path.exists(f"{prefix}.cds") else 0,
            "output_pep_size": os.path.getsize(f"{prefix}.pep") if os.path.exists(f"{prefix}.pep") else 0,
        })
    
    return results


def benchmark_python_longorfs(
    transcripts_file: str, 
    output_dir: str,
    min_protein_length: int = 100,
    complete_orfs_only: bool = False,
    top_strand_only: bool = False
) -> Dict[str, Any]:
    """Benchmark the Python TransDecoder.LongOrfs implementation"""
    results = {
        "implementation": "python",
        "phase": "longorfs",
        "start_time": time.time(),
        "input_file": transcripts_file,
        "input_size": os.path.getsize(transcripts_file),
        "num_sequences": len(read_fasta(transcripts_file)),
    }
    
    # Create output directory
    py_output_dir = os.path.join(output_dir, "py_output")
    os.makedirs(py_output_dir, exist_ok=True)
    
    # Run Python implementation
    memory_before = get_memory_usage()
    start_time = time.time()
    
    try:
        # Read transcripts
        transcripts = read_fasta(transcripts_file)
        
        # Initialize extractor
        extractor = LongOrfsExtractor(
            min_protein_length=min_protein_length,
            complete_orfs_only=complete_orfs_only,
            top_strand_only=top_strand_only
        )
        
        # Extract ORFs
        orfs_by_transcript = extractor.process_transcripts(transcripts)
        
        # Write output files
        gff3_file, cds_file, pep_file = extractor.write_output_files(
            orfs_by_transcript, py_output_dir, "longest_orfs"
        )
        
        success = True
        retcode = 0
    except Exception as e:
        print(f"Error in Python LongOrfs: {e}")
        success = False
        retcode = 1
    
    end_time = time.time()
    memory_after = get_memory_usage()
    
    # Collect results
    results.update({
        "execution_time": end_time - start_time,
        "return_code": retcode,
        "memory_usage": memory_after - memory_before,
        "end_time": end_time,
        "success": success
    })
    
    # Get output file sizes
    if success:
        results.update({
            "output_gff3_size": os.path.getsize(gff3_file) if os.path.exists(gff3_file) else 0,
            "output_cds_size": os.path.getsize(cds_file) if os.path.exists(cds_file) else 0,
            "output_pep_size": os.path.getsize(pep_file) if os.path.exists(pep_file) else 0,
        })
    
    return results


def benchmark_perl_predict(
    transcripts_file: str, 
    output_dir: str,
    retain_long_orfs_length: int = 1000000,
    single_best_only: bool = False,
    pfam_hits_file: Optional[str] = None,
    blastp_hits_file: Optional[str] = None
) -> Dict[str, Any]:
    """Benchmark the Perl TransDecoder.Predict implementation"""
    results = {
        "implementation": "perl",
        "phase": "predict",
        "start_time": time.time(),
        "input_file": transcripts_file,
        "input_size": os.path.getsize(transcripts_file),
        "num_sequences": len(read_fasta(transcripts_file)),
    }
    
    # Build command
    perl_bin = Path(__file__).parent.parent.parent / "TransDecoder.Predict"
    cmd = f"{perl_bin} -t {transcripts_file} --output_dir {output_dir} --retain_long_orfs_length {retain_long_orfs_length} --no_refine_starts"
    
    if single_best_only:
        cmd += " --single_best_only"
    if pfam_hits_file:
        cmd += f" --retain_pfam_hits {pfam_hits_file}"
    if blastp_hits_file:
        cmd += f" --retain_blastp_hits {blastp_hits_file}"
    
    # Run command
    memory_before = get_memory_usage()
    stdout, stderr, retcode, execution_time = run_command(cmd)
    memory_after = get_memory_usage()
    
    # Collect results
    results.update({
        "execution_time": execution_time,
        "return_code": retcode,
        "memory_usage": memory_after - memory_before,
        "end_time": time.time(),
        "command": cmd,
        "success": retcode == 0
    })
    
    # Get output file sizes
    base_name = os.path.basename(transcripts_file)
    final_prefix = os.path.join(output_dir, f"{base_name}.transdecoder")
    
    if os.path.exists(f"{final_prefix}.gff3"):
        results.update({
            "output_gff3_size": os.path.getsize(f"{final_prefix}.gff3"),
            "output_bed_size": os.path.getsize(f"{final_prefix}.bed") if os.path.exists(f"{final_prefix}.bed") else 0,
            "output_cds_size": os.path.getsize(f"{final_prefix}.cds") if os.path.exists(f"{final_prefix}.cds") else 0,
            "output_pep_size": os.path.getsize(f"{final_prefix}.pep") if os.path.exists(f"{final_prefix}.pep") else 0,
        })
    
    return results


def benchmark_python_predict(
    transcripts_file: str, 
    output_dir: str,
    retain_long_orfs_length: int = 1000000,
    single_best_only: bool = False,
    pfam_hits_file: Optional[str] = None,
    blastp_hits_file: Optional[str] = None
) -> Dict[str, Any]:
    """Benchmark the Python TransDecoder.Predict implementation"""
    results = {
        "implementation": "python",
        "phase": "predict",
        "start_time": time.time(),
        "input_file": transcripts_file,
        "input_size": os.path.getsize(transcripts_file),
        "num_sequences": len(read_fasta(transcripts_file)),
    }
    
    # Create output directory
    py_output_dir = os.path.join(output_dir, "py_output")
    os.makedirs(py_output_dir, exist_ok=True)
    
    # Run Python implementation
    memory_before = get_memory_usage()
    start_time = time.time()
    
    try:
        # Initialize predictor
        predictor = TransDecoderPredictor(
            transcripts_file=transcripts_file,
            output_dir=py_output_dir,
            retain_long_orfs_mode="strict",
            retain_long_orfs_length=retain_long_orfs_length,
            top_orfs_train=500,
            genetic_code="universal"
        )
        
        # Run pipeline
        predictor.run_pipeline(
            pfam_hits_file=pfam_hits_file,
            blastp_hits_file=blastp_hits_file,
            single_best_only=single_best_only,
            no_refine_starts=True
        )
        
        success = True
        retcode = 0
    except Exception as e:
        print(f"Error in Python Predict: {e}")
        success = False
        retcode = 1
    
    end_time = time.time()
    memory_after = get_memory_usage()
    
    # Collect results
    results.update({
        "execution_time": end_time - start_time,
        "return_code": retcode,
        "memory_usage": memory_after - memory_before,
        "end_time": end_time,
        "success": success
    })
    
    # Get output file sizes
    if success:
        base_name = os.path.basename(transcripts_file)
        final_prefix = os.path.join(py_output_dir, f"{base_name}.transdecoder")
        
        if os.path.exists(f"{final_prefix}.gff3"):
            results.update({
                "output_gff3_size": os.path.getsize(f"{final_prefix}.gff3"),
                "output_bed_size": os.path.getsize(f"{final_prefix}.bed") if os.path.exists(f"{final_prefix}.bed") else 0,
                "output_cds_size": os.path.getsize(f"{final_prefix}.cds") if os.path.exists(f"{final_prefix}.cds") else 0,
                "output_pep_size": os.path.getsize(f"{final_prefix}.pep") if os.path.exists(f"{final_prefix}.pep") else 0,
            })
    
    return results


def run_benchmark(
    transcripts_file: str,
    output_dir: str,
    min_protein_length: int = 100,
    complete_orfs_only: bool = False,
    top_strand_only: bool = False,
    retain_long_orfs_length: int = 1000000,
    single_best_only: bool = False,
    pfam_hits_file: Optional[str] = None,
    blastp_hits_file: Optional[str] = None,
    run_predict: bool = True,
    num_runs: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    """Run benchmarks for Perl and Python implementations"""
    results = {
        "perl_longorfs": [],
        "python_longorfs": [],
        "perl_predict": [],
        "python_predict": []
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run each benchmark multiple times
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}:")
        
        # Create separate output directory for this run
        run_dir = os.path.join(output_dir, f"run_{run+1}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Run LongOrfs benchmarks
        print("  Running Perl LongOrfs...")
        perl_longorfs_result = benchmark_perl_longorfs(
            transcripts_file, run_dir, min_protein_length, complete_orfs_only, top_strand_only
        )
        results["perl_longorfs"].append(perl_longorfs_result)
        
        print("  Running Python LongOrfs...")
        python_longorfs_result = benchmark_python_longorfs(
            transcripts_file, run_dir, min_protein_length, complete_orfs_only, top_strand_only
        )
        results["python_longorfs"].append(python_longorfs_result)
        
        # Run Predict benchmarks if requested
        if run_predict:
            # Only run predict if LongOrfs succeeded
            if perl_longorfs_result["success"]:
                print("  Running Perl Predict...")
                perl_predict_result = benchmark_perl_predict(
                    transcripts_file, run_dir, retain_long_orfs_length, single_best_only,
                    pfam_hits_file, blastp_hits_file
                )
                results["perl_predict"].append(perl_predict_result)
            else:
                print("  Skipping Perl Predict (LongOrfs failed)")
            
            if python_longorfs_result["success"]:
                print("  Running Python Predict...")
                python_predict_result = benchmark_python_predict(
                    transcripts_file, run_dir, retain_long_orfs_length, single_best_only,
                    pfam_hits_file, blastp_hits_file
                )
                results["python_predict"].append(python_predict_result)
            else:
                print("  Skipping Python Predict (LongOrfs failed)")
    
    return results


def save_benchmark_results(results: Dict[str, List[Dict[str, Any]]], output_file: str) -> None:
    """Save benchmark results to a file"""
    # Flatten results for CSV output
    flat_results = []
    
    for impl_phase, runs in results.items():
        for run_idx, run_data in enumerate(runs):
            run_data["run_number"] = run_idx + 1
            run_data["impl_phase"] = impl_phase
            flat_results.append(run_data)
    
    # Save to CSV
    fieldnames = set()
    for result in flat_results:
        fieldnames.update(result.keys())
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=sorted(fieldnames))
        writer.writeheader()
        for result in flat_results:
            writer.writerow(result)
    
    print(f"Results saved to {output_file}")


def generate_benchmark_report(results_file: str, output_dir: str) -> None:
    """Generate a report from benchmark results"""
    # Read results
    df = pd.read_csv(results_file)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Split by phase
    longorfs_df = df[df['phase'] == 'longorfs']
    predict_df = df[df['phase'] == 'predict']
    
    # Create summary tables
    longorfs_summary = longorfs_df.groupby('implementation').agg({
        'execution_time': ['mean', 'std', 'min', 'max'],
        'memory_usage': ['mean', 'std', 'min', 'max'],
        'success': 'mean'
    })
    
    predict_summary = predict_df.groupby('implementation').agg({
        'execution_time': ['mean', 'std', 'min', 'max'],
        'memory_usage': ['mean', 'std', 'min', 'max'],
        'success': 'mean'
    })
    
    # Save summary tables
    longorfs_summary.to_csv(os.path.join(output_dir, 'longorfs_summary.csv'))
    predict_summary.to_csv(os.path.join(output_dir, 'predict_summary.csv'))
    
    # Create plots
    
    # 1. Execution time comparison
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    longorfs_time = longorfs_df.groupby('implementation')['execution_time'].mean()
    longorfs_time.plot(kind='bar', yerr=longorfs_df.groupby('implementation')['execution_time'].std())
    plt.title('LongOrfs Execution Time')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=0)
    
    plt.subplot(1, 2, 2)
    if not predict_df.empty:
        predict_time = predict_df.groupby('implementation')['execution_time'].mean()
        predict_time.plot(kind='bar', yerr=predict_df.groupby('implementation')['execution_time'].std())
        plt.title('Predict Execution Time')
        plt.ylabel('Time (seconds)')
        plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'execution_time.png'))
    
    # 2. Memory usage comparison
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    longorfs_mem = longorfs_df.groupby('implementation')['memory_usage'].mean() / (1024 * 1024)  # Convert to MB
    longorfs_mem.plot(kind='bar', yerr=longorfs_df.groupby('implementation')['memory_usage'].std() / (1024 * 1024))
    plt.title('LongOrfs Memory Usage')
    plt.ylabel('Memory (MB)')
    plt.xticks(rotation=0)
    
    plt.subplot(1, 2, 2)
    if not predict_df.empty:
        predict_mem = predict_df.groupby('implementation')['memory_usage'].mean() / (1024 * 1024)
        predict_mem.plot(kind='bar', yerr=predict_df.groupby('implementation')['memory_usage'].std() / (1024 * 1024))
        plt.title('Predict Memory Usage')
        plt.ylabel('Memory (MB)')
        plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'memory_usage.png'))
    
    # 3. Success rate
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    longorfs_success = longorfs_df.groupby('implementation')['success'].mean() * 100
    longorfs_success.plot(kind='bar')
    plt.title('LongOrfs Success Rate')
    plt.ylabel('Success Rate (%)')
    plt.xticks(rotation=0)
    plt.ylim(0, 105)
    
    plt.subplot(1, 2, 2)
    if not predict_df.empty:
        predict_success = predict_df.groupby('implementation')['success'].mean() * 100
        predict_success.plot(kind='bar')
        plt.title('Predict Success Rate')
        plt.ylabel('Success Rate (%)')
        plt.xticks(rotation=0)
        plt.ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'success_rate.png'))
    
    # Create HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>TransDecoder Benchmark Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .chart {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>TransDecoder Benchmark Report</h1>
        
        <h2>LongOrfs Phase</h2>
        <table>
            <tr>
                <th>Implementation</th>
                <th>Execution Time (s)</th>
                <th>Memory Usage (MB)</th>
                <th>Success Rate (%)</th>
            </tr>
            <tr>
                <td>Perl</td>
                <td>{longorfs_df[longorfs_df['implementation'] == 'perl']['execution_time'].mean():.2f} ± {longorfs_df[longorfs_df['implementation'] == 'perl']['execution_time'].std():.2f}</td>
                <td>{longorfs_df[longorfs_df['implementation'] == 'perl']['memory_usage'].mean() / (1024 * 1024):.2f} ± {longorfs_df[longorfs_df['implementation'] == 'perl']['memory_usage'].std() / (1024 * 1024):.2f}</td>
                <td>{longorfs_df[longorfs_df['implementation'] == 'perl']['success'].mean() * 100:.1f}%</td>
            </tr>
            <tr>
                <td>Python</td>
                <td>{longorfs_df[longorfs_df['implementation'] == 'python']['execution_time'].mean():.2f} ± {longorfs_df[longorfs_df['implementation'] == 'python']['execution_time'].std():.2f}</td>
                <td>{longorfs_df[longorfs_df['implementation'] == 'python']['memory_usage'].mean() / (1024 * 1024):.2f} ± {longorfs_df[longorfs_df['implementation'] == 'python']['memory_usage'].std() / (1024 * 1024):.2f}</td>
                <td>{longorfs_df[longorfs_df['implementation'] == 'python']['success'].mean() * 100:.1f}%</td>
            </tr>
        </table>
        
        <div class="chart">
            <img src="execution_time.png" alt="Execution Time Comparison" style="width: 100%;">
        </div>
        
        <div class="chart">
            <img src="memory_usage.png" alt="Memory Usage Comparison" style="width: 100%;">
        </div>
        
        <div class="chart">
            <img src="success_rate.png" alt="Success Rate Comparison" style="width: 100%;">
        </div>
        
        <h2>Summary</h2>
        <p>
            <b>Performance Comparison:</b> Python vs Perl
        </p>
        <ul>
            <li>Execution Time: Python is {longorfs_df[longorfs_df['implementation'] == 'perl']['execution_time'].mean() / max(0.001, longorfs_df[longorfs_df['implementation'] == 'python']['execution_time'].mean()):.2f}x {("faster" if longorfs_df[longorfs_df['implementation'] == 'perl']['execution_time'].mean() > longorfs_df[longorfs_df['implementation'] == 'python']['execution_time'].mean() else "slower")} than Perl for LongOrfs</li>
            <li>Memory Usage: Python uses {longorfs_df[longorfs_df['implementation'] == 'perl']['memory_usage'].mean() / max(0.001, longorfs_df[longorfs_df['implementation'] == 'python']['memory_usage'].mean()):.2f}x {("less" if longorfs_df[longorfs_df['implementation'] == 'perl']['memory_usage'].mean() > longorfs_df[longorfs_df['implementation'] == 'python']['memory_usage'].mean() else "more")} memory than Perl for LongOrfs</li>
        </ul>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, 'report.html'), 'w') as f:
        f.write(html)
    
    print(f"Report generated in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Perl and Python TransDecoder implementations")
    parser.add_argument("--input", required=True, help="Input transcripts file")
    parser.add_argument("--output_dir", default="benchmark_results", help="Output directory for benchmark results")
    parser.add_argument("--min_protein_length", type=int, default=100, help="Minimum protein length")
    parser.add_argument("--complete_orfs_only", action="store_true", help="Only report complete ORFs")
    parser.add_argument("--top_strand_only", action="store_true", help="Only analyze top strand")
    parser.add_argument("--retain_long_orfs_length", type=int, default=1000000, help="Minimum ORF length to retain in strict mode")
    parser.add_argument("--single_best_only", action="store_true", help="Retain only single best ORF per transcript")
    parser.add_argument("--pfam_hits", help="Pfam domain table hits file")
    parser.add_argument("--blastp_hits", help="BlastP hits file")
    parser.add_argument("--skip_predict", action="store_true", help="Skip predict phase benchmarks")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument("--generate_report", action="store_true", help="Generate benchmark report")
    parser.add_argument("--report_input", help="Input CSV file for report generation")
    parser.add_argument("--report_output", help="Output directory for report generation")
    
    args = parser.parse_args()
    
    # Generate report only
    if args.generate_report and args.report_input and args.report_output:
        generate_benchmark_report(args.report_input, args.report_output)
        sys.exit(0)
    
    # Run benchmarks
    results = run_benchmark(
        transcripts_file=args.input,
        output_dir=args.output_dir,
        min_protein_length=args.min_protein_length,
        complete_orfs_only=args.complete_orfs_only,
        top_strand_only=args.top_strand_only,
        retain_long_orfs_length=args.retain_long_orfs_length,
        single_best_only=args.single_best_only,
        pfam_hits_file=args.pfam_hits,
        blastp_hits_file=args.blastp_hits,
        run_predict=not args.skip_predict,
        num_runs=args.runs
    )
    
    # Save results
    results_file = os.path.join(args.output_dir, "benchmark_results.csv")
    save_benchmark_results(results, results_file)
    
    # Generate report if requested
    if args.generate_report:
        report_output = args.report_output or os.path.join(args.output_dir, "report")
        generate_benchmark_report(results_file, report_output)