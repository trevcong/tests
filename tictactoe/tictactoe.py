"""
Tic Tac Toe Player
"""
import copy
import math

X = "X"
O = "O"
EMPTY = None
currentplayer = 'X'

def copyboard(board):
    return copy.deepcopy(board)


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    x = 0
    o = 0

    for row in board:
        x += row.count(X)
        o += row.count(O)

        if x > o:
            return O
        else:
            return X

def actions(board):
    avaliblelocations = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == EMPTY:
                avaliblelocations.append((i, j))
    return avaliblelocations
    


def result(board, action):
    moves = actions(board)
    if action not in moves:
        raise ValueError
    newboard = copyboard(board)
    moveaction = list(action)
    newboard[moveaction[0][moveaction[1]]]
    return newboard
    


def winner(board):
    """
    Returns the winner of the game if there is one, otherwise None.
    """
    # Check rows for winner
    for row in board:
        if row[0] == row[1] == row[2] and row[0] is not EMPTY:
            return row[0]

    # Check columns for winner
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] is not EMPTY:
            return board[0][col]

    # Check diagonals for winner
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not EMPTY:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not EMPTY:
        return board[0][2]

    return None



def terminal(board):
    if winner(board) is not None:
        return True
    
    for row in board:
        if EMPTY in row:
            return False
    
    return False


def utility(board):
    win = winner(board)
    if win == "X":
        return 1
    elif win == "O":
        return -1
    else:
        return 0 


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    raise NotImplementedError
