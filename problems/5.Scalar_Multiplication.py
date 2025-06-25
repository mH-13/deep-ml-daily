#Scalar Multiplication of a Matrix

def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    result = []
    for row in matrix:
        for element in row:
            new_row = scalar * element
            result.append(new_row)
            
    return result
  
  
# documentationdef __doc__():
    """    This function performs scalar multiplication on a given matrix.
    Parameters:
    matrix (list[list[int|float]]): A 2D list representing the matrix.
    scalar (int|float): The scalar value to multiply with each element of the matrix.
    Returns:
    list[list[int|float]]: A 2D list representing the resulting matrix after scalar multiplication.
    
    Scalar multiplication is used to:
    Scale gradients
    Control learning rate effects
    Normalize weights
    Apply regularization (like multiplying loss or weight penalty)
    """
    pass 
  
# Example usage
if __name__ == "__main__":  
    matrix = [[1, 2, 3], [4, 5, 6]]
    scalar = 2
    print(scalar_multiply(matrix, scalar))  # Output: [[2, 4, 6], [8, 10, 12]]

    matrix = [[1.5, 2.5], [3.5, 4.5]]
    scalar = 3
    print(scalar_multiply(matrix, scalar))  # Output: [[4.5, 7.5], [10.5, 13.5]]


# using machine learning to solve this
import numpy as np
def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    np_matrix = np.array(matrix)
    result = scalar * np_matrix
    
    return result.tolist()