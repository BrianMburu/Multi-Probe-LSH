## Multi-Probe LSH

This project contains a Python implementation of the Multi-Probe LSH algorithm, which is a method for approximate nearest neighbor search. The algorithm builds an index of known embeddings and labels, then uses this index to efficiently find the closest neighbor of a given query embedding.

### Technologies

The project is written in Python and uses the following libraries:

- Time: For timing the model when testing its performance.
- Annoy: For building the approximate nearest neighbor index.

### Installation

To use the MultiProbLSH class, you will need to install the NumPy and Annoy libraries. You can install these using pip:

- `pip install annoy`

### Usage

To use the MultiProbLSH class, you can import it into your Python code:

```python
from multi_prob_lsh import MultiProbLSH

# Create a MultiProbLSH object
lsh = MultiProbLSH(n_trees=10)

# Initialize the index with known embeddings
known_embeddings = [...] # A list of NumPy arrays
known_labels = [...] # A list of labels corresponding to the embeddings
lsh.initialize(known_embeddings)

# Find the closest neighbor of a query embedding
query_embedding = [...] # A NumPy array
closest_label, closest_distance = lsh.get_closest_neighbour(query_embedding, known_labels)

# Test the performance of the index on a test set
testX = [...] # A list of NumPy arrays
testy = [...] # A list of labels corresponding to the test embeddings
accuracy, mis_labeled, total_time = lsh.test_performance(testX, testy, known_labels)
```

### Examples

You can find some examples of how to use the MultiProbLSH class in the examples directory.

### Time complexity

The time complexity of the Multi-Probe LSH algorithm depends on the number of trees used to build the index (n_trees) and the number of probes used to search the index (n_probes). The time complexity of building the index is O(N log N), where N is the number of known embeddings. The time complexity of querying the index is O(log N) per query for a fixed number of n_probes. However, increasing the value of n_probes can result in slower query times.

### References:

- I. S. Dhillon, Y. Guan, and B. Kulis. "Efficient kernels for supervised graph learning." Proceedings of the Eleventh International Conference on Artificial Intelligence and Statistics (AISTATS-2007), pp. 97-104, 2007.
- A. Andoni and P. Indyk. "Near-optimal hashing algorithms for approximate nearest neighbor in high dimensions." Communications of the ACM, Vol. 51, No. 1, pp. 117-122, 2008.
- A. Andoni and P. Indyk. "Near-optimal hashing algorithms for approximate nearest neighbor in high dimensions." SIAM Journal on Computing, Vol. 39, No. 5, pp. 2108-2131, 2010.
