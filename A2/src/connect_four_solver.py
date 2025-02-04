import math
import random
import numpy as np

# Board dimensions for Connect Four
ROW_COUNT = 6
COLUMN_COUNT = 7

def create_board():
    """
    Creates an empty Connect Four board (numpy 2D array).

    Returns:
    np.ndarray:
        A 2D numpy array of shape (ROW_COUNT, COLUMN_COUNT) filled with zeros (float).
    """
    # TODO: implement
    board =  np.zeros((ROW_COUNT, COLUMN_COUNT), float)
    return board


def drop_piece(board, row, col, piece):
    """
    Places a piece (1 or 2) at the specified (row, col) position on the board.

    Parameters:
    board (np.ndarray): The current board, shape (ROW_COUNT, COLUMN_COUNT).
    row (int): The row index where the piece should be placed.
    col (int): The column index where the piece should be placed.
    piece (int): The piece identifier (1 or 2).
    
    Returns:
    None. The 'board' is modified in-place. Do NOT return a new board!
    """
    # TODO: implement
    #if is_valid_location:
    board[row][col] = piece

def is_valid_location(board, col):
    """
    Checks if dropping a piece in 'col' is valid (column not full).

    Parameters:
    board (np.ndarray): The current board.
    col (int): The column index to check.

    Returns:
    bool: True if it's valid to drop a piece in this column, False otherwise.
    """
    # TODO: implement
    if board[0][col] == 0:
        return True
    return False


def get_next_open_row(board, col):
    """
    Gets the next open row in the given column.

    Parameters:
    board (np.ndarray): The current board.
    col (int): The column index to search.

    Returns:
    int: The row index of the lowest empty cell in this column.
    """
    # TODO: implement
    empty_row = -1
    for i in range(ROW_COUNT):
        if board[i][col] == 1 or board[i][col] == 2:
            empty_row = i
            break
    if empty_row == -1:
        return 0
    else:
        return empty_row - 1

def winning_move(board, piece):
    """
    Checks if the board state is a winning state for the given piece.

    Parameters:
    board (np.ndarray): The current board.
    piece (int): The piece identifier (1 or 2).

    Returns:
    bool: True if 'piece' has a winning 4 in a row, False otherwise.
    This requires checking horizontally, vertically, and diagonally.
    """
    # TODO: implement
    row, col = ROW_COUNT-1, COLUMN_COUNT-1
    for i in range(row-1):
        for j in range(col-1):
            if board[row][col] == piece:
                if winning_diagonally(board, (i,j), piece):
                    return True
                if winning_horizontally(board, (i,j), piece):
                    return True
                if winning_vertically(board, (i,j), piece):
                    return True
    return False

def winning_horizontally(board, place: tuple, piece):
    row,col = place
    counter = 0
    while col < COLUMN_COUNT:
        if board[row][col] == piece:
            counter += 1
        else:
            counter = 0
        if counter == 4:
            return True
        col += 1

    row,col = place
    while col >= 0:
        if board[row][col] == piece:
            counter += 1
        else:
            counter = 0
        if counter == 4:
            return True
        col -= 1
    return False

def winning_vertically(board, place: tuple, piece):
    row,col = place
    counter = 0
    while row < ROW_COUNT:
        if board[row][col] == piece:
            counter += 1
        else:
            counter = 0
        if counter == 4:
            return True
        row += 1

    row,col = place
    while row >= 0:
        if board[row][col] == piece:
            counter += 1
        else:
            counter = 0
        if counter == 4:
            return True
        row -= 1
    return False

def winning_diagonally(board, place: tuple, piece):
    row,col = place
    counter = 0
    while row < ROW_COUNT and col < COLUMN_COUNT:
        if board[row][col] == piece:
            counter += 1
        else:
            counter = 0
        if counter == 4:
            return True
        row += 1
        col += 1
    
    row,col = place
    while row >= 0 and col >= 0:
        if board[row][col] == piece:
            counter += 1
        else:
            counter = 0
        if counter == 4:
            return True
        row -= 1
        col -= 1
    
    row,col = place
    while row < ROW_COUNT and col >= 0:
        if board[row][col] == piece:
            counter += 1
        else:
            counter = 0
        if counter == 4:
            return True
        row += 1
        col -= 1
    
    row,col = place
    while row >= 0 and col < COLUMN_COUNT:
        if board[row][col] == piece:
            counter += 1
        else:
            counter = 0
        if counter == 4:
            return True
        row -= 1
        col += 1
    return False                                       


def get_valid_locations(board):
    """
    Returns a list of columns that are valid to drop a piece.

    Parameters:
    board (np.ndarray): The current board.

    Returns:
    list of int: The list of column indices that are not full.
    """
    # TODO: implement
    not_full = []
    for i in range(COLUMN_COUNT):
        if board[0][i] == 0:
            not_full.append(i)
    return not_full


def is_terminal_node(board):
    """
    Checks if the board is in a terminal state:
      - Either player has won
      - Or no valid moves remain

    Parameters:
    board (np.ndarray): The current board.

    Returns:
    bool: True if the game is over, False otherwise.
    """
    # TODO: implement
    if winning_move(board, 1):
        return True

    if winning_move(board, 2):
        return True
    
    if get_valid_locations(board) == []:
        return True

    return False


def score_position(board, piece):
    """
    Evaluates the board for the given piece.
    (Already implemented to highlight center-column preference.)

    Parameters:
    board (np.ndarray): The current board.
    piece (int): The piece identifier (1 or 2).

    Returns:
    int: Score favoring the center column. 
         (This can be extended with more advanced heuristics.)
    """
    # This is already done for you; no need to modify
    # The heuristic here scores the center column higher, which means
    # it prefers to play in the center column.
    score = 0
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3
    return score


def minimax(board, depth, alpha, beta, maximizingPlayer):
    """
    Performs minimax with alpha-beta pruning to choose the best column.

    Parameters:
    board (np.ndarray): The current board.
    depth (int): Depth to which the minimax tree should be explored.
    alpha (float): Alpha for alpha-beta pruning.
    beta (float): Beta for alpha-beta pruning.
    maximizingPlayer (bool): Whether it's the maximizing player's turn.

    Returns:
    tuple:
        - (column (int or None), score (float)):
          column: The chosen column index (None if no moves).
          score: The heuristic score of the board state.
    """
    # TODO: implement
    


if __name__ == "__main__":
    # Simple debug scenario
    # Example usage: create a board, drop some pieces, then call minimax
    example_board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    print("Debug: Created an empty Connect Four board.\n")
    print(example_board)

    # TODO: Students can test their own logic here once implemented, e.g.:
    # drop_piece(example_board, some_row, some_col, 1)
    # col, score = minimax(example_board, depth=4, alpha=-math.inf, beta=math.inf, maximizingPlayer=True)
    # print("Chosen column:", col, "Score:", score)
