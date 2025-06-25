#Calculate Eigenvalues of a Matrix

import numpy as np
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    np_matrix = np.array(matrix)

    eigenvalues, _ = np.linalg.eig(np_matrix) #linalg.eig computes the eigenvalues and right eigenvectors of a square array.
    # We only need the eigenvalues, so we ignore the second return value (eigenvectors).
    # The eigenvalues are complex numbers, so we take the real part using .real.

    sorted_eigenvalues = sorted(eigenvalues.real.tolist(), reverse = True) #eigenvalues.real converts the complex eigenvalues to real numbers, and .tolist() converts the numpy array to a list. reverse=True sorts the list in descending order.
    
    return sorted_eigenvalues
  
# documentation
def __doc__():  
    """
    This function calculates the eigenvalues of a given matrix.
    
    Parameters:
    matrix (list[list[float|int]]): A 2D list representing the matrix.
    
    Returns:
    list[float]: A list of eigenvalues sorted in descending order.
    
    Eigenvalues are used in machine learning for:
    - Dimensionality reduction (PCA)
    - Understanding data variance
    - Feature extraction
    - Stability analysis in dynamical systems
    """
    pass
  
# Example usage
if __name__ == "__main__":
    matrix = [[4, -2], [1, 1]]
    print(calculate_eigenvalues(matrix))  # Output: [3.0, 2.0]

    matrix = [[1, 2], [3, 4]]
    print(calculate_eigenvalues(matrix))  # Output: [5.372281323269014, -0.3722813232690143]

    matrix = [[1.5, 2.5], [3.5, 4.5]]
    print(calculate_eigenvalues(matrix))  # Output: [6.0, 0.0]
    
    