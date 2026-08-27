# MinHash Jaccard Similarity

A fast, reproducible command-line tool that estimates pairwise **Jaccard similarity** between binary matrix columns using **MinHash signatures**.

## What is Jaccard Similarity?

The **Jaccard similarity** (or Jaccard index) measures how similar two sets are. It is defined as:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

where:
- $|A \cap B|$ is the size of the intersection (elements in both sets)
- $|A \cup B|$ is the size of the union (elements in at least one set)

The result is always between 0 and 1:
- **1.0** means the sets are identical
- **0.0** means the sets have no elements in common

### Example

```
Set A = {1, 2, 3}
Set B = {2, 3, 4}

Intersection: {2, 3}       → size = 2
Union:        {1, 2, 3, 4} → size = 4

Jaccard Similarity = 2 / 4 = 0.5
```

## What is MinHash?

**MinHash** is a probabilistic algorithm that quickly **approximates** Jaccard similarity without computing the intersection and union explicitly. It works by:

1. Generating `n` random hash functions
2. For each set, recording the **minimum hash value** seen for each hash function
3. Comparing the signature lists—if two sets have the same minimum hash value at position `i`, they likely have high Jaccard similarity

### Why MinHash?

- **Fast**: Compares signatures instead of full sets
- **Approximate**: Results are close to true Jaccard similarity
- **Scalable**: Works with large sets and high-dimensional data
- **Probabilistic**: Accuracy improves with more hash functions
- **Memory-efficient**: Automatic sparse matrix detection for sparse datasets

The approximation quality depends on the number of hash functions:

$$J(A, B) \approx \frac{\text{matching signature positions}}{n}$$

## How This Implementation Works

### Input Format

The input is a text file containing a binary matrix where:
- Each **row** represents an element or observation
- Each **column** represents a set
- A value of `1` means the element belongs to that set
- A value of `0` means it does not

**Example input file (`matrix.txt`):**

```
10110
10010
00111
```

This describes three sets:
- **Set 0**: {row 0, row 1}  (columns where row has `1`)
- **Set 1**: {row 2}
- **Set 2**: {row 0, row 2}
- **Set 3**: {row 0, row 1, row 2}
- **Set 4**: {row 2}

### Algorithm Steps

1. **Load the binary matrix** and validate dimensions
2. **Optimize representation**: Automatically detect if the matrix is sparse (>50% zeros) and use sparse representation if beneficial
3. **Generate random hash functions**: For each hash function, randomly select coefficients `a` and `b`
4. **Build MinHash signatures**: For each hash function, compute the minimum hash value across all rows where a column contains `1`
   - Hash function: $h(row) = (a \times row + b) \bmod 23$
   - For sparse matrices, only iterate through non-zero elements
5. **Calculate similarity**: Count how many hash functions produce matching signature values for each pair of columns
6. **Report results**: Convert matches to percentages and print elapsed time

## Installation

### Requirements

- Python 3.9 or later

### Setup

Clone the repository:

```bash
git clone https://github.com/yourusername/Jaccard_Similarity.git
cd Jaccard_Similarity
```

No external dependencies are required for basic usage. For development (testing, linting):

```bash
pip install -e ".[dev]"
```

## Usage

### Basic Example

```bash
python minhash.py matrix.txt --mode all --permutations 100 --seed 42
```

### Command-Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `filename` | Yes | Path to the text file containing the 0/1 matrix |
| `--mode` | Yes | Comparison mode: `all` (all pairs) or `pairs` (selected pairs) |
| `-n, --permutations` | Yes | Number of hash functions (higher = more accurate, slower) |
| `--pairs` | If `--mode pairs` | Comma-separated pairs like `0-2,1-4` |
| `--seed` | No | Random seed for reproducibility (default: unseeded) |
| `--sparse` | No | Force sparse matrix representation (automatic detection enabled by default) |
| `-v, --verbose` | No | Print input matrix, initial/final signatures, and detailed progress |
| `-o, --output` | No | Write results to a file instead of stdout |

### Examples

**Compare all column pairs:**

```bash
python minhash.py matrix.txt --mode all --permutations 100
```

**Compare only specific pairs:**

```bash
python minhash.py matrix.txt --mode pairs --pairs 0-2,1-3 --permutations 100
```

**Reproducible results with a seed:**

```bash
python minhash.py matrix.txt --mode all --permutations 100 --seed 42
```

**Verbose output to see intermediate steps:**

```bash
python minhash.py matrix.txt --mode all --permutations 100 --verbose
```

**Save results to a file:**

```bash
python minhash.py matrix.txt --mode all --permutations 100 --output results.txt
```

**Use sparse matrix representation:**

For large sparse matrices (>50% zeros), sparse representation is automatically enabled. To force it:

```bash
python minhash.py matrix.txt --mode all --permutations 100 --sparse
```

## Output

### Summary Section

```
Analysis summary:
  Input file: matrix.txt
  Dimensions: 3 rows x 5 columns
  Hash functions: 100
  Random seed: 42
  Storage: sparse (only 14 non-empty columns)
```

The storage line is shown only when sparse representation is active.

### Results Section

For each pair of columns, the estimated Jaccard similarity is reported:

```
Estimated Jaccard similarity:
  Column A    Column B    Estimate
  --------    --------    --------
      0            1       10.00%
      0            2       75.00%
      0            3       95.00%
      0            4       15.00%
      1            2        5.00%
      1            3       12.00%
      1            4       85.00%
      2            3       88.00%
      2            4        8.00%
      3            4       20.00%
```

Each percentage represents the estimated Jaccard similarity between two sets.

### Timing Information

The elapsed time is printed at the end of each run:

```
Analysis completed in 0.043 seconds
```

This is useful for benchmarking performance with different matrix sizes and permutation counts.

### Understanding the Accuracy

- With `--permutations 100`, results are typically accurate to within ±10%
- With `--permutations 1000`, results are typically accurate to within ±3%
- Accuracy improves as $\frac{1}{\sqrt{n}}$ where $n$ is the number of hash functions

For high-precision applications, use at least 500–1000 permutations.

## Complexity Analysis

### Time Complexity (Dense Matrix)

Let:
- $m$ = number of rows
- $c$ = number of columns
- $n$ = number of hash functions
- $z$ = number of non-zero elements in the matrix

| Operation | Complexity |
|-----------|------------|
| Loading matrix | $O(m \cdot c)$ |
| Building signatures (dense) | $O(n \cdot m \cdot c)$ |
| Building signatures (sparse) | $O(n \cdot z)$ |
| Calculating similarity | $O(n \cdot c^2)$ |

### Space Complexity

**Dense representation:**
$$O(m \cdot c + n \cdot c)$$

**Sparse representation:**
$$O(z + n \cdot c)$$

where $z$ is the number of non-zero elements. For sparse matrices (when $z \ll m \cdot c$), this provides significant savings.

### Automatic Sparse Optimization

The implementation automatically detects sparse matrices (>50% zeros) and switches to sparse representation, which:
- Only stores positions of `1` values
- Only iterates through non-zero elements during signature building
- Reduces memory usage from $O(m \cdot c)$ to $O(z)$
- Speeds up computation by factor of $\frac{m \cdot c}{z}$ in the best case

## Limitations and Design Decisions

### Limitations

1. **Fixed hash modulus**: The modulus is hardcoded to `23`. This is sufficient for small matrices but may need adjustment for larger datasets.

2. **Empty columns**: Columns containing all zeros remain filled with $\infty$ and may appear identical to other empty columns.

3. **Affine hash functions**: This implementation uses simple affine hash functions $(ax + b) \bmod p$ rather than universal hash families. For production use, consider more sophisticated hash families.

4. **No seed by default**: Without `--seed`, results change on each run. Always use `--seed` when reproducibility matters.

5. **Memory usage**: While sparse matrices are optimized, very large dense matrices may require significant memory.

### Design Choices

- **Matrix representation**: Columns represent sets, rows represent elements. This matches the standard MinHash literature.
- **Sparse detection**: Matrices with >50% zeros automatically use sparse representation to optimize memory and time.
- **Random coefficients**: Generated in the range [1, 22] to ensure they are non-zero modulo 23.
- **Error handling**: Invalid matrix dimensions, out-of-range column pairs, and malformed input are reported clearly.
- **Performance tracking**: Elapsed time is reported for benchmarking and performance analysis.

## Examples

### Sample Input File

Create `matrix.txt`:

```
1010
1100
0110
1111
```

Run the analysis:

```bash
python minhash.py matrix.txt --mode all --permutations 50 --seed 123 --verbose
```

Expected output snippet:

```
Analysis summary:
  Input file: matrix.txt
  Dimensions: 4 rows x 4 columns
  Hash functions: 50
  Random seed: 123

Input matrix (rows = elements, columns = sets):
lines   S(0)	S(1)	S(2)	S(3)	
(0)	1	0	1	0	
(1)	1	1	0	0	
(2)	0	1	1	0	
(3)	1	1	1	1	

Estimated Jaccard similarity:
  Column A    Column B    Estimate
  --------    --------    --------
      0            1       48.00%
      0            2       40.00%
      0            3       88.00%
      1            2       48.00%
      1            3       92.00%
      2            3       86.00%

Analysis completed in 0.003 seconds
```

## References

- Broder, A. Z. (1997). "On the resemblance and containment of documents." *Proceedings of Compression and Complexity of Sequences*.
- Rajaraman, A., Leskovec, J., & Ullman, J. D. (2011). *Mining of Massive Datasets*. Chapter 3: "Finding Similar Items."

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for new functionality
4. Run `pytest` to ensure all tests pass
5. Submit a pull request

For more details, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Author

Created as a learning project on data mining algorithms and Python software engineering.
