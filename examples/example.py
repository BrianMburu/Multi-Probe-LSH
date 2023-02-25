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