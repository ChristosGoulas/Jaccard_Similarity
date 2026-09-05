"""Tests for the MinHash Jaccard similarity estimator."""

import argparse
import math
import random
import tempfile
from pathlib import Path

import pytest

from minhash import (
    calculate_similarity,
    create_signature_matrix,
    matrix_sparsity,
    parse_pairs,
    positive_integer,
    read_matrix,
    to_sparse_matrix,
    update_signature,
    update_signature_sparse,
    build_signatures,
)


class TestPositiveInteger:
    """Tests for positive_integer argument parser."""

    def test_valid_positive_integer(self):
        """Test parsing valid positive integers."""
        assert positive_integer("1") == 1
        assert positive_integer("42") == 42
        assert positive_integer("1000") == 1000

    def test_zero_raises_error(self):
        """Test that zero raises ArgumentTypeError."""
        with pytest.raises(argparse.ArgumentTypeError):
            positive_integer("0")

    def test_negative_raises_error(self):
        """Test that negative numbers raise ArgumentTypeError."""
        with pytest.raises(argparse.ArgumentTypeError):
            positive_integer("-1")
        with pytest.raises(argparse.ArgumentTypeError):
            positive_integer("-100")

    def test_non_integer_raises_error(self):
        """Test that non-integers raise ArgumentTypeError."""
        with pytest.raises(argparse.ArgumentTypeError):
            positive_integer("abc")
        with pytest.raises(argparse.ArgumentTypeError):
            positive_integer("3.14")
        with pytest.raises(argparse.ArgumentTypeError):
            positive_integer("")


class TestReadMatrix:
    """Tests for reading binary matrices from files."""

    def test_read_valid_matrix(self):
        """Test reading a valid binary matrix."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("10\n01\n")
            fname = f.name

        try:
            matrix = read_matrix(fname)
            assert matrix == [[1, 0], [0, 1]]
            assert len(matrix) == 2
            assert len(matrix[0]) == 2
        finally:
            Path(fname).unlink()

    def test_read_larger_matrix(self):
        """Test reading a larger matrix."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("10110\n10010\n00111\n")
            fname = f.name

        try:
            matrix = read_matrix(fname)
            assert len(matrix) == 3
            assert len(matrix[0]) == 5
            assert matrix[0] == [1, 0, 1, 1, 0]
            assert matrix[1] == [1, 0, 0, 1, 0]
            assert matrix[2] == [0, 0, 1, 1, 1]
        finally:
            Path(fname).unlink()

    def test_empty_file_raises_error(self):
        """Test that empty file raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            fname = f.name

        try:
            with pytest.raises(ValueError, match="empty"):
                read_matrix(fname)
        finally:
            Path(fname).unlink()

    def test_inconsistent_row_length_raises_error(self):
        """Test that inconsistent row lengths raise ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("101\n10\n")
            fname = f.name

        try:
            with pytest.raises(ValueError, match="row 2"):
                read_matrix(fname)
        finally:
            Path(fname).unlink()

    def test_invalid_character_raises_error(self):
        """Test that non-binary characters raise ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("101\n102\n")
            fname = f.name

        try:
            with pytest.raises(ValueError, match="row 2"):
                read_matrix(fname)
        finally:
            Path(fname).unlink()

    def test_nonexistent_file_raises_error(self):
        """Test that nonexistent file raises OSError."""
        with pytest.raises(OSError):
            read_matrix("/nonexistent/path/file.txt")

    def test_single_row_matrix(self):
        """Test reading a single-row matrix."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("10101\n")
            fname = f.name

        try:
            matrix = read_matrix(fname)
            assert len(matrix) == 1
            assert matrix[0] == [1, 0, 1, 0, 1]
        finally:
            Path(fname).unlink()

    def test_single_column_matrix(self):
        """Test reading a single-column matrix."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("1\n0\n1\n")
            fname = f.name

        try:
            matrix = read_matrix(fname)
            assert len(matrix) == 3
            assert all(len(row) == 1 for row in matrix)
        finally:
            Path(fname).unlink()


class TestMatrixSparsity:
    """Tests for matrix sparsity calculation."""

    def test_dense_matrix(self):
        """Test sparsity of all-ones matrix."""
        matrix = [[1, 1], [1, 1]]
        assert matrix_sparsity(matrix) == 0.0

    def test_sparse_matrix(self):
        """Test sparsity of mostly-zeros matrix."""
        matrix = [[1, 0], [0, 0]]
        assert matrix_sparsity(matrix) == 0.75

    def test_all_zeros_matrix(self):
        """Test sparsity of all-zeros matrix."""
        matrix = [[0, 0], [0, 0]]
        assert matrix_sparsity(matrix) == 1.0

    def test_half_sparse_matrix(self):
        """Test sparsity of 50% zeros matrix."""
        matrix = [[1, 0], [1, 0]]
        assert matrix_sparsity(matrix) == 0.5


class TestSparseMatrix:
    """Tests for sparse matrix conversion."""

    def test_convert_dense_to_sparse(self):
        """Test converting dense matrix to sparse representation."""
        matrix = [[1, 0], [0, 1]]
        sparse = to_sparse_matrix(matrix)
        assert sparse == {0: {0}, 1: {1}}

    def test_sparse_matrix_larger(self):
        """Test sparse conversion with larger matrix."""
        matrix = [[1, 0, 1], [0, 1, 0], [1, 1, 0]]
        sparse = to_sparse_matrix(matrix)
        assert sparse == {
            0: {0, 2},
            1: {1, 2},
            2: {0},
        }

    def test_all_zeros_sparse(self):
        """Test sparse conversion of all-zeros matrix."""
        matrix = [[0, 0], [0, 0]]
        sparse = to_sparse_matrix(matrix)
        assert sparse == {}

    def test_single_column_sparse(self):
        """Test sparse conversion of single column."""
        matrix = [[1], [0], [1]]
        sparse = to_sparse_matrix(matrix)
        assert sparse == {0: {0, 2}}


class TestSignatureMatrix:
    """Tests for MinHash signature matrix operations."""

    def test_create_empty_signature_matrix(self):
        """Test creating an empty signature matrix."""
        sig_matrix = create_signature_matrix(5, 3)
        assert len(sig_matrix) == 5
        assert all(len(row) == 3 for row in sig_matrix)
        assert all(all(val == math.inf for val in row) for row in sig_matrix)

    def test_update_signature_single_row(self):
        """Test updating signature with a single row."""
        matrix = [[1, 0, 1]]
        sig_matrix = create_signature_matrix(1, 3)
        update_signature(0, 1, matrix, sig_matrix, 0, 3)
        assert sig_matrix[0] == [1, math.inf, 1]

    def test_update_signature_multiple_rows(self):
        """Test that minimum permuted rank is kept."""
        matrix = [[1, 0], [1, 0]]
        sig_matrix = create_signature_matrix(1, 2)
        # First row has permuted rank 1
        update_signature(0, 1, matrix, sig_matrix, 0, 2)
        assert sig_matrix[0][0] == 1
        # Second row has permuted rank 2 (greater than 1, not updated)
        update_signature(1, 2, matrix, sig_matrix, 0, 2)
        assert sig_matrix[0][0] == 1  # Still 1 (minimum)

    def test_large_row_count_no_modulo_collision(self):
        """Test matrices with >23 rows have no modulo collision artifacts."""
        # 100 rows, single column with all 1s
        matrix = [[1] for _ in range(100)]
        sig_matrix = create_signature_matrix(10, 1)
        rng = random.Random(42)
        build_signatures(matrix, 10, sig_matrix, rng, use_sparse=False)
        # Minimum permutation value across 100 items must be 0 for a full column of 1s
        for row in sig_matrix:
            assert row[0] == 0

    def test_build_signatures_basic(self):
        """Test building signatures with dense matrix."""
        matrix = [[1, 0], [0, 1]]
        sig_matrix = create_signature_matrix(1, 2)
        rng = random.Random(42)
        build_signatures(matrix, 1, sig_matrix, rng, use_sparse=False)
        # Signatures should be populated
        assert sig_matrix[0][0] != math.inf
        assert sig_matrix[0][1] != math.inf

    def test_build_signatures_sparse(self):
        """Test building signatures with sparse matrix."""
        matrix = [[1, 0], [0, 1]]
        sparse = to_sparse_matrix(matrix)
        sig_matrix = create_signature_matrix(1, 2)
        rng = random.Random(42)
        build_signatures(sparse, 1, sig_matrix, rng, use_sparse=True)
        # Should produce same result
        assert sig_matrix[0][0] != math.inf
        assert sig_matrix[0][1] != math.inf

    def test_build_signatures_reproducible(self):
        """Test that same seed produces identical signatures."""
        matrix = [[1, 0, 1], [1, 1, 0], [0, 1, 1]]
        
        sig_matrix_1 = create_signature_matrix(50, 3)
        rng_1 = random.Random(123)
        build_signatures(matrix, 50, sig_matrix_1, rng_1, use_sparse=False)
        
        sig_matrix_2 = create_signature_matrix(50, 3)
        rng_2 = random.Random(123)
        build_signatures(matrix, 50, sig_matrix_2, rng_2, use_sparse=False)
        
        assert sig_matrix_1 == sig_matrix_2


class TestSimilarityCalculation:
    """Tests for Jaccard similarity estimation."""

    def test_identical_columns(self):
        """Test that identical columns have 100% similarity."""
        # Create identical signatures
        sig_matrix = [
            [1, 1, 2],
            [3, 3, 5],
            [7, 7, 11],
        ]
        sim_matrix = calculate_similarity(sig_matrix)
        # Columns 0 and 1 have all matching signatures (3/3 = 100%)
        similarity = (sim_matrix[1][0] / 3) * 100
        assert similarity == 100.0

    def test_completely_different_columns(self):
        """Test that completely different columns have 0% similarity."""
        sig_matrix = [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
        sim_matrix = calculate_similarity(sig_matrix)
        # Columns 0 and 1 have no matching signatures (0/3 = 0%)
        similarity = (sim_matrix[1][0] / 3) * 100
        assert similarity == 0.0

    def test_empty_columns_do_not_match(self):
        """Test that empty columns do not create false signature matches."""
        sig_matrix = [
            [math.inf, math.inf],
            [math.inf, math.inf],
            [math.inf, math.inf],
        ]
        sim_matrix = calculate_similarity(sig_matrix)
        assert sim_matrix[1][0] == 0

    def test_partial_similarity(self):
        """Test partial similarity between columns."""
        sig_matrix = [
            [1, 1],
            [2, 3],
            [4, 4],
        ]
        sim_matrix = calculate_similarity(sig_matrix)
        # Columns 0 and 1 match in positions 0 and 2 (2/3 = 66.67%)
        similarity = (sim_matrix[1][0] / 3) * 100
        assert abs(similarity - 66.67) < 0.1

    def test_similarity_matrix_symmetric(self):
        """Test that similarity matrix is symmetric."""
        sig_matrix = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        sim_matrix = calculate_similarity(sig_matrix)
        # Check symmetry
        for i in range(len(sim_matrix)):
            for j in range(len(sim_matrix)):
                assert sim_matrix[i][j] == sim_matrix[j][i]


class TestParsePairs:
    """Tests for parsing column pair arguments."""

    def test_single_pair(self):
        """Test parsing a single column pair."""
        pairs = parse_pairs("0-1", 5)
        assert pairs == [(0, 1)]

    def test_multiple_pairs(self):
        """Test parsing multiple column pairs."""
        pairs = parse_pairs("0-2,1-4,2-3", 5)
        assert pairs == [(0, 2), (1, 4), (2, 3)]

    def test_pairs_with_spaces(self):
        """Test parsing pairs with spaces."""
        pairs = parse_pairs("0 - 1, 2 - 3", 5)
        assert pairs == [(0, 1), (2, 3)]

    def test_pair_out_of_range_raises_error(self):
        """Test that out-of-range pairs raise ValueError."""
        with pytest.raises(ValueError, match="outside the column range"):
            parse_pairs("0-10", 5)

    def test_same_column_pair_raises_error(self):
        """Test that same column pair raises ValueError."""
        with pytest.raises(ValueError, match="two different columns"):
            parse_pairs("0-0", 5)

    def test_invalid_format_raises_error(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError, match="expected the format i-j"):
            parse_pairs("0-1-2", 5)

    def test_non_integer_pair_raises_error(self):
        """Test that non-integer pairs raise ValueError."""
        with pytest.raises(ValueError, match="must be integers"):
            parse_pairs("a-b", 5)


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_full_pipeline_dense(self):
        """Test complete analysis with dense matrix."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("10\n01\n11\n")
            fname = f.name

        try:
            matrix = read_matrix(fname)
            sig_matrix = create_signature_matrix(10, 2)
            rng = random.Random(42)
            build_signatures(matrix, 10, sig_matrix, rng, use_sparse=False)
            sim_matrix = calculate_similarity(sig_matrix)
            
            # Should have non-inf similarities
            assert sig_matrix[0][0] != math.inf
            assert sig_matrix[0][1] != math.inf
            assert len(sim_matrix) == 2
            assert len(sim_matrix[0]) == 2
        finally:
            Path(fname).unlink()

    def test_full_pipeline_sparse(self):
        """Test complete analysis with sparse matrix."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            # Very sparse: mostly zeros
            f.write("10000\n00100\n00010\n")
            fname = f.name

        try:
            matrix = read_matrix(fname)
            sparsity = matrix_sparsity(matrix)
            assert sparsity > 0.5  # Is sparse
            
            sparse = to_sparse_matrix(matrix)
            sig_matrix = create_signature_matrix(10, 5)
            rng = random.Random(42)
            build_signatures(sparse, 10, sig_matrix, rng, use_sparse=True)
            sim_matrix = calculate_similarity(sig_matrix)
            
            assert len(sim_matrix) == 5
            assert len(sim_matrix[0]) == 5
        finally:
            Path(fname).unlink()

    def test_large_permutation_count(self):
        """Test with large number of permutations."""
        matrix = [[1, 0, 1], [1, 1, 0], [0, 1, 1]]
        sig_matrix = create_signature_matrix(1000, 3)
        rng = random.Random(42)
        build_signatures(matrix, 1000, sig_matrix, rng, use_sparse=False)
        
        # All signatures should be populated
        for row in sig_matrix:
            for val in row:
                assert val != math.inf

    def test_edge_case_all_zeros_column(self):
        """Test handling of columns with all zeros."""
        matrix = [[1, 0], [1, 0], [1, 0]]
        sig_matrix = create_signature_matrix(5, 2)
        rng = random.Random(42)
        build_signatures(matrix, 5, sig_matrix, rng, use_sparse=False)
        
        # Column 0 should be populated
        assert sig_matrix[0][0] != math.inf
        # Column 1 should remain as infinity (no ones)
        assert sig_matrix[0][1] == math.inf
