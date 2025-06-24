#concept: 
#Problem Statement: Transpose a Matrix# Given a matrix, write a function to transpose it.
# The transpose of a matrix is obtained by swapping its rows and columns.
# For example, the transpose of the matrix [[1, 2, 3], [4, 5, 6]] is [[1, 4], [2, 5], [3, 6]].

def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    rows = len(a)
    cols = len(a[0])

    transposed_matrix = []
    
    for j in range(cols):
        updated_row = []
        for i in range(rows): 
            updated_row.append(a[i][j])
        transposed_matrix.append(updated_row)
    return transposed_matrix

# documentation
def __doc__():
    """
    This function transposes a given matrix.
    
    Parameters:
    a (list[list[int|float]]): A 2D list representing the matrix.
    Returns:
    list[list[int|float]]: A 2D list representing the transposed matrix.
    """
    pass  
  
# Example usage
if __name__ == "__main__":  
    a = [[1, 2, 3], [4, 5, 6]]
    print(transpose_matrix(a))  # Output: [[1, 4], [2, 5], [3, 6]]

    b = [[1.5, 2.5], [3.5, 4.5]]
    print(transpose_matrix(b))  # Output: [[1.5, 3.5], [2.5, 4.5]]
  
  
##using machine learning to solve this
#a.T transposes the matrix, swapping rows and columns.
#tolist() converts the numpy array to a list of lists, which is the expected output format.
import numpy as np
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    np_matrix = np.array(a)
    transposed_matrix = np_matrix.T
    
    return transposed_matrix.tolist()  # [[1, 4], [2, 5], [3, 6]]
