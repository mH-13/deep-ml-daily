def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
      # Return a list where each element is the dot product of a row of 'a' with 'b'.
      # If the number of columns in 'a' does not match the length of 'b', return -1.
      if len(a[0]) != len(b):
          return -1

      res= []
      
      for row in a:
          dotproduct = 0
          for i in range(len(a[0])):
              dotproduct += row[i] * b[i]
          res.append(dotproduct)
      return res
    
# documentation
def __doc__():
    """
    This function computes the dot product of each row of a matrix with a vector.
    
    Parameters:
    a (list[list[int|float]]): A 2D list representing the matrix.
    b (list[int|float]): A 1D list representing the vector.
    
    Returns:
    list[int|float]: A list where each element is the dot product of a row of 'a' with 'b'.
                      Returns -1 if the number of columns in 'a' does not match the length of 'b'.
    """
    pass  
  
# Example usage
if __name__ == "__main__":
    a = [[1, 2, 3], [4, 5, 6]]
    b = [7, 8, 9]
    print(matrix_dot_vector(a, b))  # Output: [50, 122]

    a = [[1, 2], [3, 4]]
    b = [5, 6, 7]
    print(matrix_dot_vector(a, b))  # Output: -1
    a = [[1.5, 2.5], [3.5, 4.5]]
    b = [5.5, 6.5]
    print(matrix_dot_vector(a, b))  # Output: [23.0, 53.0] 
    
#using machine learning  to solve this

