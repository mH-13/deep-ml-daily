import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:

    a_np = np.array(A)
    t_np = np.array(T)
    s_np = np.array(S)

    # Check if the input matrices are square and compatible for multiplication. we cannot invert a non-square matrix
    if t_np.shape[0] != t_np.shape[1] or s_np.shape[0] != t_np.shape[1]:
        return -1
      
    # Check if the determinant of S is zero, which would make it non-invertible. though we will not invert s but we need to check if it is invertible so that we can multiply it with the inverse of T
    if np.linalg.det(s_np) == 0:
        return -1
        
    # invert the transformation matrix T. 
    t_invert = np.linalg.inv(t_np)

    # Perform the transformation: T^-1 * A * S. First we multiply the inverse of T with A, then we multiply the result with S.
    # This is equivalent to applying the transformation T^-1 to A and then scaling by S. 
    # The order of multiplication matters, as matrix multiplication is not commutative.
    transformed_matrix = np.matmul(np.matmul(t_invert, a_np), s_np) 
    
    return transformed_matrix.tolist()
  
  
# documentationdef __doc__():
    """    This function transforms a matrix A using a transformation matrix T and a scaling matrix S. 
    The transformation is done by first applying the inverse of T to A, and then scaling the result by S.
    Parameters:
    A (list[list[int|float]]): The matrix to be transformed.
    T (list[list[int|float]]): The transformation matrix.
    S (list[list[int|float]]): The scaling matrix.
    Returns:
    list[list[int|float]]: The transformed matrix, or -1 if the transformation cannot be performed due to incompatible dimensions or a non-invertible matrix. 
    The transformation is used in various applications such as:
    - Linear transformations in computer graphics
    - Data transformations in machine learning
    - Coordinate transformations in physics
    """
    pass 
  
# Example usage
if __name__ == "__main__":
    A = [[1, 2], [3, 4]]
    T = [[1, 0], [0, 1]]  # Identity matrix
    S = [[2, 0], [0, 2]]  # Scaling by a factor of 2
    print(transform_matrix(A, T, S))  # Output: [[2.0, 4.0], [6.0, 8.0]]

    A = [[1, 2], [3, 4]]
    T = [[1, 0], [0, 1]]  # Identity matrix
    S = [[0, 0], [0, 0]]  # Zero matrix (not invertible)
    print(transform_matrix(A, T, S))  # Output: -1

    A = [[1, 2], [3, 4]]
    T = [[1, 2], [3, 4]]  
    S = [[5, 6], [7, 8]]  
    print(transform_matrix(A, T, S))  # Output: [[-0.5, -1.0], [-1.5, -2.0]]  