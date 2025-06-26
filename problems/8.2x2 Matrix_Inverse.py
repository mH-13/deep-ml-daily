import numpy as np
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:

    # Check if the input is a 2x2 matrix
    if len(matrix) != 2 or len(matrix[0]) != 2 or len(matrix[1]) != 2:
        return None

    a,b = matrix[0]
    c,d = matrix[1]

    det = a*d - b*c
    if det == 0:
        return None
    
    # Calculate the inverse using the formula for a 2x2 matrix
    # The inverse of a 2x2 matrix [[a, b], [c, d]] is given by:
    # 1/det * [[d, -b], [-c, a]]
    inv_det = 1/det
    inverse_matrix = [
        [d*inv_det,  -b*inv_det],
        [-c*inv_det,  a*inv_det]
    ]
    
    return inverse_matrix
  
# documentation
def __doc__():
    """
    This function calculates the inverse of a 2x2 matrix.
    
    Parameters:
    matrix (list[list[float]]): A 2D list representing the 2x2 matrix.
    
    Returns:
    list[list[float]]: The inverse of the matrix if it exists, otherwise None.
    
    The inverse of a matrix is used in various applications such as:
    - Solving systems of linear equations
    - Computer graphics transformations
    - Machine learning algorithms
    """
    pass 


# Example usageif __name__ == "__main__":
    matrix = [[4, 2], [3, 1]]
    print(inverse_2x2(matrix))  # Output: [[0.25, -0.5], [-0.75, 2.0]]
    matrix = [[1, 2], [3, 4]]
    print(inverse_2x2(matrix))  # Output: [[-2.0, 1.0], [1.5, -0.5]]
    matrix = [[1, 2], [2, 4]]
    print(inverse_2x2(matrix))  # Output: None (matrix is singular)