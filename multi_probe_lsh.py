import annoy 
import time

class MultiProbLSH:
    """
    Multi-Probe LSH algorithm class.
    """

    def __init__(self, index=None, n_trees=10, include_distances=True):
        """
        Constructor for MultiProbLSH class.

        Args:
            index (annoy.AnnoyIndex): Pre-initialized Annoy index.
            n_trees (int): Number of trees to build the index with.
            include_distances (bool): Whether to include distance information in nearest neighbor queries.
        """
        self.n_trees = n_trees
        self.include_distances = include_distances
        self.index = index

    def initialize(self, known_embeddings):
        """
        Initializes the Multi-Probe LSH index with a set of known embeddings.

        Args:
            known_embeddings (list of np.array): List of known embeddings to use to build the index.
        """
        index = annoy.AnnoyIndex(len(known_embeddings[0]), 'angular')

        # Index the known embeddings and labels
        for i, _ in enumerate(known_embeddings):
            index.add_item(i, known_embeddings[i])

        index.build(n_trees=self.n_trees)
        self.index = index

    def get_closest_neighbour(self, query_embedding, known_labels):
        """
        Finds the closest neighbor of a given query embedding in the index.

        Args:
            query_embedding (np.array): Query embedding to find the closest neighbor for.
            known_labels (list): List of labels corresponding to the known embeddings.

        Returns:
            tuple: A tuple containing the label of the closest neighbor and the distance to that neighbor.
        """
        closest_index, closest_distance = self.index.get_nns_by_vector(query_embedding, 1, search_k=-1,
                                                                       include_distances=self.include_distances)
        closest_label = known_labels[closest_index[0]]

        return closest_label, closest_distance[0]

    def test_performance(self, testX, testy, known_labels):
        """
        Tests the performance of the Multi-Probe LSH index on a given test set.

        Args:
            testX (list of np.array): Test set embeddings.
            testy (list): List of true labels for the test set.
            known_labels (list): List of labels corresponding to the known embeddings.

        Returns:
            tuple: A tuple containing the accuracy, number of mislabeled samples, and the total query time.
        """
        mis_labeled = 0
        start_time = time.time()

        for i, _ in enumerate(testX):
            closest_label, closest_distance = self.get_closest_neighbour(testX[i], known_labels)
            if testy[i] != closest_label:
                mis_labeled += 1

        end_time = time.time()
        total_time = end_time - start_time
        accuracy = 100 - (mis_labeled / len(known_labels))

        return accuracy, mis_labeled, total_time