"""
Tests for the utility functions in the TransDecoder package.
"""

import os
import tempfile
import pytest
from transdecoder.utils import (
    reverse_complement, translate_sequence, compute_gc_content
)

def test_reverse_complement():
    """Test the reverse complement function"""
    assert reverse_complement("ATGC") == "GCAT"
    assert reverse_complement("NNNNN") == "NNNNN"
    assert reverse_complement("ATGCATGC") == "GCATGCAT"
    assert reverse_complement("atgc") == "gcat"

def test_translate_sequence():
    """Test the translation function"""
    # Test standard translation
    assert translate_sequence("ATGCCC") == "MP"
    assert translate_sequence("ATGCCCTGA") == "MP*"
    
    # Test different genetic codes
    assert translate_sequence("ATGCCC", "universal") == "MP"
    
    # Test partial codons
    assert translate_sequence("ATGCC") == "MP"

def test_compute_gc_content():
    """Test GC content calculation"""
    assert compute_gc_content("ATGC") == 0.5
    assert compute_gc_content("AAAA") == 0.0
    assert compute_gc_content("CCCC") == 1.0
    assert compute_gc_content("ATGCATGC") == 0.5
    
    # Test mixed case
    assert compute_gc_content("atgc") == 0.5