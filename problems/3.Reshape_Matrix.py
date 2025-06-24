import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
    np_matrix = np.array(a)
    
    if np_matrix.size != new_shape[0] * new_shape[1]:
      return [] # Return an empty list if the total number of elements does not match the new shape
    reshaped_matrix = np_matrix.reshape(new_shape)
    
    return reshaped_matrix.tolist()

# documentation
def __doc__():
    """
    This function reshapes a given matrix to a new shape.
    
    Parameters:
    a (list[list[int|float]]): A 2D list representing the matrix.
    new_shape (tuple[int, int]): A tuple representing the new shape of the matrix.
    
    Returns:
    list[list[int|float]]: A 2D list representing the reshaped matrix.
    """
    pass
  
# Example usage
if __name__ == "__main__":
    a = [[1, 2, 3], [4, 5, 6]]
    new_shape = (3, 2)
    print(reshape_matrix(a, new_shape))  # Output: [[1, 2], [3, 4], [5, 6]]

    b = [[1.5, 2.5], [3.5, 4.5]]
    new_shape = (2, 2)
    print(reshape_matrix(b, new_shape))  # Output: [[1.5, 2.5], [3.5, 4.5]]
    