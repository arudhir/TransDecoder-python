# TransDecoder-python

A Python implementation of the [TransDecoder](https://github.com/TransDecoder/TransDecoder) pipeline for identifying coding regions in transcripts.

## About

TransDecoder-python is a Python port of the original Perl-based [TransDecoder](https://github.com/TransDecoder/TransDecoder) software. It identifies likely coding sequences in transcripts from RNA-Seq assemblies, implementing the same algorithms as the original tool but in a more accessible Python codebase.

The system identifies long open reading frames (ORFs) within transcript sequences and scores them using log-likelihood scores based on a hexamer Markov model and other features, following the original TransDecoder methodology.

## Installation

```bash
pip install transdecoder
```

Or install directly from the repository:

```bash
git clone https://github.com/arudhir/TransDecoder-python.git
cd TransDecoder-python
pip install -e .
```

## Usage

### Simple Example

```python
from transdecoder import longorfs, predict

# Run the LongOrfs process on a FASTA file
longorfs.extract_long_orfs("transcripts.fasta", output_dir="output_dir", min_length=100)

# Run the Predict process to score and select the best ORFs
predict.predict_coding_regions("transcripts.fasta", output_dir="output_dir")
```

For more detailed examples, see the `examples` directory.

## Benchmarks

The Python implementation has been benchmarked against the original Perl implementation. See the benchmark results in the `benchmark_results` directory.

## Testing

Run the tests with:

```bash
make test
```

Or directly with pytest:

```bash
pytest tests/
```

## Algorithm Documentation

TransDecoder uses a multi-step approach to identify coding regions:

1. **Long ORFs Extraction**: Identifies all ORFs that meet a minimum length requirement.
2. **Markov Model Training**: Builds a hexamer frequency model based on the longest ORFs.
3. **ORF Scoring**: Calculates log-likelihood scores for each ORF based on the hexamer frequencies.
4. **Best ORF Selection**: Selects the highest-scoring ORF for each transcript.

See the original [TransDecoder documentation](https://github.com/TransDecoder/TransDecoder/wiki) for more detailed algorithm descriptions.

## License

MIT License (same as the original TransDecoder)

## Acknowledgments

This is a port of the original [TransDecoder](https://github.com/TransDecoder/TransDecoder) software developed by Brian Haas et al. at the Broad Institute.
