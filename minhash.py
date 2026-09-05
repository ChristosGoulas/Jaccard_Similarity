"""Estimate Jaccard similarity using MinHash signatures.

A fast, reproducible command-line tool that estimates pairwise Jaccard similarity
between binary matrix columns using MinHash signatures.

Version:
    1.0.0

License:
    MIT
"""

__version__ = "1.0.0"
__author__ = "Christos Goulas"
__license__ = "MIT"

import argparse
import logging
import math
import random
import sys
import time
from contextlib import redirect_stdout
from typing import Optional, Sequence, Union

BinaryMatrix = list[list[int]]
SignatureMatrix = list[list[Union[int, float]]]
SparseMatrix = dict[int, set[int]]  # Maps column index to set of row indices with 1s

SPARSITY_THRESHOLD = 0.5  # Use sparse representation if > 50% zeros

LOGGER = logging.getLogger(__name__)


def positive_integer(value: str) -> int:
    """Parse a strictly positive command-line integer."""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Estimate pairwise Jaccard similarity with MinHash."
    )
    parser.add_argument(
        "filename",
        help="path to a text file containing a 0/1 matrix",
    )
    parser.add_argument(
        "-n",
        "--permutations",
        type=positive_integer,
        required=True,
        help="number of hash functions used for each signature",
    )
    parser.add_argument(
        "--mode",
        choices=("all", "pairs"),
        required=True,
        help="compare all columns or only selected pairs",
    )
    parser.add_argument(
        "--pairs",
        help="comma-separated column pairs, for example 1-3,2-4",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="write progress information to stderr",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="write the final report to a file instead of stdout",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed the random generator for reproducible results",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="force sparse matrix representation (automatically detected for large sparse matrices)",
    )
    return parser


def parse_pairs(pair_input: str, column_count: int) -> list[tuple[int, int]]:
    """Parse and validate column pairs supplied to the CLI."""
    pairs: list[tuple[int, int]] = []
    for pair_text in pair_input.split(","):
        values = pair_text.strip().split("-")
        if len(values) != 2:
            raise ValueError(f"invalid pair {pair_text!r}; expected the format i-j")
        try:
            column_i, column_j = (int(value.strip()) for value in values)
        except ValueError as error:
            raise ValueError(f"invalid pair {pair_text!r}; indices must be integers") from error
        if not (0 <= column_i < column_count) or not (0 <= column_j < column_count):
            raise ValueError(
                f"pair {pair_text!r} is outside the column range " f"0-{column_count - 1}"
            )
        if column_i == column_j:
            raise ValueError("a pair must contain two different columns")
        pairs.append((column_i, column_j))
    return pairs


def update_signature(
    row_index: int,
    permuted_value: int,
    matrix: BinaryMatrix,
    signature_matrix: SignatureMatrix,
    hash_index: int,
    column_count: int,
) -> None:
    """Update one MinHash signature row with the permuted value of one input row."""
    for column_index in range(column_count):
        if matrix[row_index][column_index] == 1:
            if signature_matrix[hash_index][column_index] > permuted_value:
                signature_matrix[hash_index][column_index] = permuted_value


def update_signature_sparse(
    permutation: Sequence[int],
    sparse_matrix: SparseMatrix,
    signature_matrix: SignatureMatrix,
    hash_index: int,
    column_count: int,
) -> None:
    """Update MinHash signatures using sparse matrix representation.

    For each column, iterate only through rows containing 1s.
    """
    for column_index in range(column_count):
        if column_index in sparse_matrix:
            for row_index in sparse_matrix[column_index]:
                permuted_value = permutation[row_index]
                if signature_matrix[hash_index][column_index] > permuted_value:
                    signature_matrix[hash_index][column_index] = permuted_value


def read_matrix(filename: str) -> BinaryMatrix:
    """Read a text file containing a binary matrix.

    Args:
        filename: Path to a text file with binary values (0s and 1s).

    Returns:
        A list of lists representing the binary matrix.

    Raises:
        OSError: If file cannot be opened.
        ValueError: If matrix is empty, has invalid dimensions, or contains non-binary values.
    """
    MAX_MATRIX_ELEMENTS = 100_000_000  # 100M elements as safety limit

    with open(filename) as matrix_file:
        rows = [line.rstrip("\r\n") for line in matrix_file]
    if not rows:
        raise ValueError("the matrix file is empty")
    column_count = len(rows[0])
    if column_count == 0:
        raise ValueError("the matrix must contain at least one column")

    # Validate matrix size before processing all rows
    total_elements = len(rows) * column_count
    if total_elements > MAX_MATRIX_ELEMENTS:
        raise ValueError(
            f"matrix size {len(rows)}x{column_count} ({total_elements} elements) "
            f"exceeds maximum allowed size of {MAX_MATRIX_ELEMENTS} elements"
        )

    for row_number, row in enumerate(rows, start=1):
        if len(row) != column_count:
            raise ValueError(f"row {row_number} has length {len(row)}; " f"expected {column_count}")
        if any(value not in "01" for value in row):
            raise ValueError(f"row {row_number} contains a value other than 0 or 1")
    return [[int(value) for value in row] for row in rows]


def matrix_sparsity(matrix: BinaryMatrix) -> float:
    """Calculate the proportion of zeros in the matrix."""
    if not matrix or not matrix[0]:
        return 0.0
    total_elements = len(matrix) * len(matrix[0])
    zero_count = sum(row.count(0) for row in matrix)
    return zero_count / total_elements


def to_sparse_matrix(matrix: BinaryMatrix) -> SparseMatrix:
    """Convert a dense binary matrix to sparse representation.

    Returns a dict mapping column index to set of row indices containing 1.
    """
    sparse: SparseMatrix = {}
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            if value == 1:
                if column_index not in sparse:
                    sparse[column_index] = set()
                sparse[column_index].add(row_index)
    return sparse


def print_matrix(matrix: BinaryMatrix) -> None:
    """Print a binary matrix using the script's tabular format."""
    row_count = len(matrix)
    column_count = len(matrix[0]) if matrix else 0
    print("Input matrix (rows = elements, columns = sets):")
    sys.stdout.write("lines   ")
    for column_index in range(column_count):
        sys.stdout.write(f"S({column_index})\t")
    print()
    for row_index in range(row_count):
        sys.stdout.write(f"({row_index})\t")
        for value in matrix[row_index]:
            sys.stdout.write(f"{value}\t")
        print()


def create_signature_matrix(permutation_count: int, column_count: int) -> SignatureMatrix:
    """Create an empty MinHash signature matrix.

    Args:
        permutation_count: Number of hash functions (rows in signature matrix).
        column_count: Number of columns (sets) to create signatures for.

    Returns:
        A matrix with permutation_count rows and column_count columns,
        initialized with infinity values.

    Raises:
        ValueError: If permutation_count or column_count is not positive.
    """
    if permutation_count <= 0:
        raise ValueError(f"permutation_count must be positive, got {permutation_count}")
    if column_count <= 0:
        raise ValueError(f"column_count must be positive, got {column_count}")
    return [[math.inf for _ in range(column_count)] for _ in range(permutation_count)]


def build_signatures(
    matrix: Union[BinaryMatrix, SparseMatrix],
    permutation_count: int,
    signature_matrix: SignatureMatrix,
    random_generator: random.Random,
    use_sparse: bool = False,
    row_count: Optional[int] = None,
) -> None:
    """Populate a signature matrix using random permutations of rows.

    Args:
        matrix: The binary matrix (dense) or SparseMatrix (sparse representation).
        permutation_count: Number of random permutations to generate.
        signature_matrix: The signature matrix to populate (modified in-place).
        random_generator: Random number generator instance.
        use_sparse: If True, expects matrix to be a SparseMatrix; if False, expects BinaryMatrix.
        row_count: Total number of rows (elements). Inferred from matrix if None.

    Raises:
        ValueError: If signature_matrix is invalid or row_count cannot be determined.
        TypeError: If matrix is not the expected type for the use_sparse setting.
    """
    if not signature_matrix or not signature_matrix[0]:
        raise ValueError("signature_matrix cannot be empty")

    column_count = len(signature_matrix[0])

    # Type validation based on use_sparse flag
    if use_sparse:
        if not isinstance(matrix, dict):
            raise TypeError(
                f"when use_sparse=True, matrix must be a SparseMatrix (dict), "
                f"got {type(matrix).__name__}"
            )
        sparse_matrix: SparseMatrix = matrix
        if row_count is None:
            max_row = max(
                (max(rows) for rows in sparse_matrix.values() if rows),
                default=-1,
            )
            num_rows = max_row + 1
        else:
            num_rows = row_count
    else:
        if not isinstance(matrix, list):
            raise TypeError(
                f"when use_sparse=False, matrix must be a BinaryMatrix (list), "
                f"got {type(matrix).__name__}"
            )
        binary_matrix: BinaryMatrix = matrix
        num_rows = len(binary_matrix) if row_count is None else row_count

    if num_rows == 0:
        return

    for hash_index in range(permutation_count):
        permutation = list(range(num_rows))
        random_generator.shuffle(permutation)

        if use_sparse:
            update_signature_sparse(
                permutation,
                sparse_matrix,
                signature_matrix,
                hash_index,
                column_count,
            )
        else:
            for row_index in range(num_rows):
                update_signature(
                    row_index,
                    permutation[row_index],
                    binary_matrix,
                    signature_matrix,
                    hash_index,
                    column_count,
                )


def print_signatures(signature_matrix: SignatureMatrix, title: str) -> None:
    """Print a titled MinHash signature matrix."""
    permutation_count = len(signature_matrix)
    column_count = len(signature_matrix[0]) if signature_matrix else 0
    print(f"{title} (rows = hash functions, columns = sets):")
    headers = ["Hash Function"] + [f"Set {index}" for index in range(column_count)]
    rows = [
        [str(hash_index)] + [str(value) for value in signature_matrix[hash_index]]
        for hash_index in range(permutation_count)
    ]
    widths = [len(header) for header in headers]
    for row in rows:
        for column_index, value in enumerate(row):
            widths[column_index] = max(widths[column_index], len(value))

    def format_row(row: list[str]) -> str:
        """Format one signature table row with stable column widths."""
        return " | ".join(value.rjust(widths[index]) for index, value in enumerate(row))

    print(format_row(headers))
    print("-+-".join("-" * width for width in widths))
    for hash_index in range(permutation_count):
        print(format_row(rows[hash_index]))


def calculate_similarity(
    signature_matrix: SignatureMatrix,
) -> list[list[int]]:
    """Count matching signature values for every pair of columns.

    Args:
        signature_matrix: The MinHash signature matrix where each row represents
                         a hash function and each column represents a set.

    Returns:
        A symmetric matrix where entry [i][j] contains the count of matching
        signatures between columns i and j.

    Raises:
        ValueError: If signature_matrix is empty or malformed.
    """
    if not signature_matrix or not signature_matrix[0]:
        raise ValueError("signature_matrix cannot be empty")

    permutation_count = len(signature_matrix)
    column_count = len(signature_matrix[0])

    # Validate all rows have same column count
    for row_index, row in enumerate(signature_matrix):
        if len(row) != column_count:
            raise ValueError(
                f"signature_matrix row {row_index} has {len(row)} columns, "
                f"expected {column_count}"
            )

    similarity_matrix = [[0 for _ in range(column_count)] for _ in range(column_count)]
    for hash_index in range(permutation_count):
        for column_i in range(column_count - 1, 0, -1):
            for column_j in range(column_i - 1, -1, -1):
                left_signature = signature_matrix[hash_index][column_j]
                right_signature = signature_matrix[hash_index][column_i]
                if (
                    left_signature != math.inf
                    and right_signature != math.inf
                    and left_signature == right_signature
                ):
                    similarity_matrix[column_i][column_j] += 1
                    similarity_matrix[column_j][column_i] += 1
    return similarity_matrix


def print_similarity(similarity_matrix: list[list[int]], permutation_count: int) -> None:
    """Print estimated similarity percentages for every column pair."""
    column_count = len(similarity_matrix)
    results: list[tuple[int, int, float]] = []
    for column_i in range(column_count):
        for column_j in range(column_i):
            similarity = similarity_matrix[column_i][column_j] / permutation_count * 100
            results.append((column_i, column_j, similarity))
    print_similarity_table(results)


def print_similarity_table(
    results: list[tuple[int, int, float]],
) -> None:
    """Print pairwise similarity results as an aligned text table."""
    print("Estimated Jaccard similarity:")
    print("  Column A    Column B    Estimate")
    print("  --------    --------    --------")
    for column_i, column_j, similarity in results:
        print(f"  {column_i:^8}    {column_j:^8}    {similarity:>7.2f}%")


def run_analysis(arguments: argparse.Namespace) -> int:
    """Run one configured MinHash analysis and produce its report.

    Args:
        arguments: Parsed command-line arguments with configuration.

    Returns:
        0 if analysis completed successfully, 2 if an error occurred during execution.
    """
    start_time = time.time()
    LOGGER.info("Starting MinHash analysis")
    LOGGER.info("Reading input matrix from %s", arguments.filename)
    try:
        matrix = read_matrix(arguments.filename)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    row_count = len(matrix)
    column_count = len(matrix[0]) if matrix else 0
    permutation_count = arguments.permutations
    LOGGER.info(
        "Input matrix loaded: %d rows x %d columns",
        row_count,
        column_count,
    )

    # Determine if sparse representation should be used
    use_sparse = arguments.sparse
    if not use_sparse and row_count > 0 and column_count > 0:
        sparsity = matrix_sparsity(matrix)
        if sparsity > SPARSITY_THRESHOLD:
            use_sparse = True
            LOGGER.info(
                "Matrix is %.1f%% sparse; using sparse representation",
                sparsity * 100,
            )

    data_to_process: Union[BinaryMatrix, SparseMatrix]
    if use_sparse:
        sparse_matrix = to_sparse_matrix(matrix)
        data_to_process = sparse_matrix
    else:
        data_to_process = matrix

    print("Analysis summary:")
    print(f"  Input file: {arguments.filename}")
    print(f"  Dimensions: {row_count} rows x {column_count} columns")
    print(f"  Hash functions: {permutation_count}")
    if arguments.seed is not None:
        print(f"  Random seed: {arguments.seed}")
    if use_sparse:
        print(f"  Storage: sparse (only {len(sparse_matrix)} non-empty columns)")

    if arguments.verbose and not use_sparse:
        print_matrix(matrix)

    LOGGER.info("Building %d MinHash signatures", permutation_count)
    if arguments.seed is not None:
        LOGGER.info("Using random seed %d", arguments.seed)
    random_generator = random.Random(arguments.seed)
    signature_matrix = create_signature_matrix(permutation_count, column_count)
    if arguments.verbose:
        print_signatures(signature_matrix, "Initial MinHash signature matrix")
    build_signatures(
        data_to_process,
        permutation_count,
        signature_matrix,
        random_generator,
        use_sparse=use_sparse,
        row_count=row_count,
    )
    if arguments.verbose:
        print_signatures(signature_matrix, "Generated MinHash signature matrix")

    LOGGER.info("Calculating pairwise Jaccard similarity estimates")
    similarity_matrix = calculate_similarity(signature_matrix)
    if arguments.mode == "pairs":
        try:
            pairs = parse_pairs(arguments.pairs, column_count)
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        results = []
        for column_i, column_j in pairs:
            similarity = similarity_matrix[column_i][column_j] / permutation_count * 100
            results.append((column_i, column_j, similarity))
        print_similarity_table(results)
    else:
        print_similarity(similarity_matrix, permutation_count)

    elapsed_time = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed_time:.3f} seconds")
    LOGGER.info("MinHash analysis complete (%.3f seconds)", elapsed_time)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command-line MinHash similarity estimator.

    Args:
        argv: Optional command-line arguments. If None, sys.argv[1:] is used.

    Returns:
        0 if execution completed successfully, 2 if an error occurred.
    """
    parser = create_argument_parser()
    arguments = parser.parse_args(argv)

    if arguments.mode == "pairs" and arguments.pairs is None:
        parser.error("--pairs is required when --mode is pairs")
    if arguments.mode == "all" and arguments.pairs is not None:
        parser.error("--pairs can only be used when --mode is pairs")

    logging.basicConfig(
        level=logging.INFO if arguments.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    if arguments.output is None:
        return run_analysis(arguments)

    LOGGER.info("Writing report to %s", arguments.output)
    try:
        with open(arguments.output, "w") as output_file:
            with redirect_stdout(output_file):
                return run_analysis(arguments)
    except OSError as error:
        print(f"error: could not write report: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
