import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if not matrix or not matrix[0]:
        return []

    np_matrix = np.array(matrix) # Convert the list of lists to a numpy array

    if mode == 'row':
        mean = np.mean(np_matrix, axis = 1) #axis=1 for row-wise mean
    elif mode == 'column':
        mean = np.mean(np_matrix, axis = 0) #axis=0 for column-wise mean
    else:
        return []
      
    return mean.tolist() # Convert the numpy array back to a list
  
# documentation
def __doc__():
    """
    This function calculates the mean of a matrix either by row or by column.
    
    Parameters:
    matrix (list[list[float]]): A 2D list representing the matrix.
    mode (str): 'row' to calculate the mean of each row, 'column' to calculate the mean of each column.
    
    Returns:
    list[float]: A list containing the mean values. If mode is invalid, returns an empty list.
    In ML, we use means to:

    Normalize inputs (make them centered around 0)

    Detect outliers

    Preprocess features
    """
    pass 
  
# Example usage
if __name__ == "__main__":  
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(calculate_matrix_mean(matrix, 'row'))      # Output: [2.0, 5.0, 8.0]
    print(calculate_matrix_mean(matrix, 'column'))   # Output: [4.0, 5.0, 6.0]

    matrix = [[1.5, 2.5], [3.5, 4.5]]
    print(calculate_matrix_mean(matrix, 'row'))      # Output: [2.0, 4.0]
    print(calculate_matrix_mean(matrix, 'column'))   # Output: [2.5, 3.5]