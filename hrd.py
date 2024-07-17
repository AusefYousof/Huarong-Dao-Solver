from copy import deepcopy
from heapq import heappush, heappop
import time
import argparse
import sys
 
# ====================================================================================
# =================================    Constants    ===================================
# ====================================================================================
 
char_goal = '1'
char_single = '2'
goal_coords = [1, 3]  # coords of the top left part of the 2x2 such that it is in goal state
 
 
# ====================================================================================
# =================================    Classes    ====================================
# ====================================================================================
 
 
class Piece:
    """
    This represents a piece on the Hua Rong Dao puzzle.
    """
 
    def __init__(self, is_goal, is_single, coord_x, coord_y, orientation, board):
        """
        :param is_goal: True if the piece is the goal piece and False otherwise.
        :type is_goal: bool
        :param is_single: True if this piece is a 1x1 piece and False otherwise.
        :type is_single: bool
        :param coord_x: The x coordinate of the top left corner of the piece.
        :type coord_x: int
        :param coord_y: The y coordinate of the top left corner of the piece.
        :type coord_y: int
        :param orientation: The orientation of the piece (one of 'h' or 'v')
            if the piece is a 1x2 piece. Otherwise, this is None
        :type orientation: str
        """
 
        self.is_goal = is_goal
        self.is_single = is_single
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.orientation = orientation
        self.board = board
 
    def __repr__(self):
        return '{} {} {} {} {}'.format(self.is_goal, self.is_single,
                                       self.coord_x, self.coord_y, self.orientation)
 
    def move_piece(self, direction):
        """
        Take in a direction and adjust the board accordingly and return True if a valid move was made and False if move invalid
        :param direction: Direction the piece wants to move
        :type direction: string
        :return valid_move, checkpiece_coord_x, checkpiece_coord_y: if move was valid, new coordinates of the piece
        :rtype: bool, int, int 
        """
 
        # get the empty slots on the board and lowercase direction for easy usage
        open_slots = self.board.find_empty_spaces()
        direction = direction.lower()
 
        #easy to refer to
        x = self.coord_x
        y = self.coord_y
 
        x1 = open_slots[0][0]
        y1 = open_slots[0][1]
        x2 = open_slots[1][0]
        y2 = open_slots[1][1]
 
        move_valid = False
        #will return these (dont want to deepcopy trying to return the piece because dont have to if move invalid)
        checkpiece_coord_x = x
        checkpiece_coord_y = y
 
 
        if direction == "up":
            if self.is_goal or self.orientation == 'h':  # need both empty slots above
                if x == x1 and x + 1 == x2 and y - 1 == y1 and y - 1 == y2:
                    checkpiece_coord_y = y1
                    move_valid = True
            elif self.is_single or self.orientation == 'v':  # need one slot above
                if x == x1 and y - 1 == y1:
                    checkpiece_coord_y = y1
                    move_valid = True
                elif x == x2 and y - 1 == y2:
                    checkpiece_coord_y = y2
                    move_valid = True
 
        elif direction == "down":
            if self.orientation == 'h':
                if x == x1 and x + 1 == x2 and y + 1 == y1 and y + 1 == y2: #need both slots below us
                    checkpiece_coord_y = y1
                    move_valid = True
            elif self.is_goal:
                if x == x1 and x + 1 == x2 and y + 2 == y1 and y + 2 == y2: #need both slots not exactly but below but two spots (top left = point of coords)
                    checkpiece_coord_y = y1-1
                    move_valid = True
            elif self.is_single:
                if x == x1 and y + 1 == y1:
                    checkpiece_coord_y = y1
                    move_valid = True
                elif x == x2 and y + 1 == y2:
                    checkpiece_coord_y = y2
                    move_valid = True
            elif self.orientation == 'v':
                if x == x1 and y + 2 == y1:
                    checkpiece_coord_y = y1-1
                    move_valid = True
                elif x == x2 and y + 2 == y2:
                    checkpiece_coord_y = y2-1
                    move_valid = True
 
        elif direction == "left":
            if self.is_goal or self.orientation == 'v':
                if x - 1 == x1 and x - 1 == x2 and y == y1 and y + 1 == y2:
                    checkpiece_coord_x = x1
                    move_valid = True
            elif self.is_single or self.orientation == 'h':
                if x - 1 == x1 and y == y1:
                    checkpiece_coord_x = x1
                    move_valid = True
                elif x - 1 == x2 and y == y2:
                    checkpiece_coord_x = x2
                    move_valid = True
 
        elif direction == "right":
            if self.orientation == 'h':
                if x + 2 == x1 and y == y1: #plus 2 again because the coords we have are of left part of h piece
                    checkpiece_coord_x = x1-1
                    move_valid = True
                elif x + 2 == x2 and y == y2:
                    checkpiece_coord_x = x2-1
                    move_valid = True
            elif self.is_goal:
                if x + 2 == x1 and x + 2 == x2 and y == y1 and y + 1 == y2:
                    checkpiece_coord_x = x1 - 1
                    move_valid = True
            elif self.is_single:
                if x + 1 == x1 and y == y1:
                    checkpiece_coord_x = x1
                    move_valid = True
                elif x + 1 == x2 and y == y2:
                    checkpiece_coord_x = x2
                    move_valid = True
            elif self.orientation == 'v':
                if x + 1 == x1 and y == y1 and x + 1 == x2 and y + 1 == y2:
                    checkpiece_coord_x = x1
                    move_valid = True
 
        
        return move_valid, checkpiece_coord_x, checkpiece_coord_y
 
        
 
 
class Board:
    """
    Board class for setting up the playing board.
    """
 
    def __init__(self, pieces):
        """
        :param pieces: The list of Pieces
        :type pieces: List[Piece]
        """
 
        self.width = 4
        self.height = 5
 
        self.pieces = pieces
 
        # self.grid is a 2-d (size * size) array automatically generated
        # using the information on the pieces when a board is being created.
        # A grid contains the symbol for representing the pieces on the board.
        self.grid = []
        self.__construct_grid()
    
    
 
  
    #hashing and comparison functions 
    def __repr__(self) -> str:
        out = ''
        for line in self.grid:
            out += ''.join(line) + '\n'
        return out
 
    def __str__(self):
        return self.__repr__()
 
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.grid == other.grid
 
    def __hash__(self) -> int:
        return hash(str(self.grid))
  
 
    def __construct_grid(self):
        """
        Called in __init__ to set up a 2-d grid based on the piece location information.
        """
 
        for i in range(self.height):
            line = []
            for j in range(self.width):
                line.append('.')
            self.grid.append(line)
 
        for piece in self.pieces:
            if piece.is_goal:
                self.grid[piece.coord_y][piece.coord_x] = char_goal
                self.grid[piece.coord_y][piece.coord_x + 1] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = char_goal
            elif piece.is_single:
                self.grid[piece.coord_y][piece.coord_x] = char_single
            else:
                if piece.orientation == 'h':
                    self.grid[piece.coord_y][piece.coord_x] = '<'
                    self.grid[piece.coord_y][piece.coord_x + 1] = '>'
                elif piece.orientation == 'v':
                    self.grid[piece.coord_y][piece.coord_x] = '^'
                    self.grid[piece.coord_y + 1][piece.coord_x] = 'v'
 
    def reconstruct_grid(self):
        """
        Reconstruct the grid after a piece has moved
        Note: A little brute forcey because im just clearing the entire grid and repopulating even though little has changed
        might seek to change it
        """
        # reset everything
        for i in range(self.height):
            for j in range(self.width):
                self.grid[i][j] = "."
 
        # repopulate (just constructing again with new x/ycoords)
        for piece in self.pieces:
            if piece.is_goal:
                self.grid[piece.coord_y][piece.coord_x] = char_goal
                self.grid[piece.coord_y][piece.coord_x + 1] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x] = char_goal
                self.grid[piece.coord_y + 1][piece.coord_x + 1] = char_goal
            elif piece.is_single:
                self.grid[piece.coord_y][piece.coord_x] = char_single
            else:
                if piece.orientation == 'h':
                    self.grid[piece.coord_y][piece.coord_x] = '<'
                    self.grid[piece.coord_y][piece.coord_x + 1] = '>'
                elif piece.orientation == 'v':
                    self.grid[piece.coord_y][piece.coord_x] = '^'
                    self.grid[piece.coord_y + 1][piece.coord_x] = 'v'
 
    def display(self):
        """
        Print out the current board.
        """
        for i, line in enumerate(self.grid):
            for ch in line:
                print(ch, end='')
            print()
 
    def find_empty_spaces(self):
        """
        Find the empty spaces in the current board.
        :return: list with two sublists with [x,y] values for the two empty slots
        :rtype: List (of 2 lists)
        """
        return [(j, i) for i, line in enumerate(self.grid) for j, ch in enumerate(line) if ch == "."]
 
 
    def get_goaltile_pos(self):
        """
        Get position of the goal tile (top left x,y value)
        :return: x,y value in the form [x,y]
        :rtype: List
        """
        for piece in self.pieces:
            if piece.is_goal:
                return [piece.coord_x, piece.coord_y]
 
 
class State:
    """
    State class wrapping a Board with some extra current state information.
    Note that State and Board are different. Board has the locations of the pieces.
    State has a Board and some extra information that is relevant to the search: 
    heuristic function, f value, current depth and parent.
    """
 
    def __init__(self, board, f, depth, parent=None):
        """
        :param board: The board of the state.
        :type board: Board
        :param f: The f value of current state.
        :type f: int
        :param depth: The depth of current state in the search tree.
        :type depth: int
        :param parent: The parent of current state.
        :type parent: Optional[State]
        """
        self.board = board
        self.f = f
        self.depth = depth
        self.parent = parent
        self.id = hash(board)  # The id for breaking ties.
 
 
    
 
    def __hash__(self):
        return self.board.__hash__()
 
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.board == other.board
 
    def __lt__(self, other):
        return self.f < other.f
    
 
 
# ====================================================================================
# =================================    Helpers    ====================================
# ====================================================================================
 
 
def read_from_file(filename):
    """
    Load initial board from a given file.
    :param filename: The name of the given file.
    :type filename: str
    :return: A loaded board
    :rtype: Board
    """
 
    puzzle_file = open(filename, "r")
 
    line_index = 0
    pieces = []
    g_found = False
 
    for line in puzzle_file:
 
        for x, ch in enumerate(line):
 
            if ch == '^':  # found vertical piece
                pieces.append(Piece(False, False, x, line_index, 'v', None))
            elif ch == '<':  # found horizontal piece
                pieces.append(Piece(False, False, x, line_index, 'h', None))
            elif ch == char_single:
                pieces.append(Piece(False, True, x, line_index, None, None))
            elif ch == char_goal:
                if g_found == False:
                    pieces.append(Piece(True, False, x, line_index, None, None))
                    g_found = True
        line_index += 1
 
    puzzle_file.close()
 
    board = Board(pieces)
 
    for piece in pieces:
        piece.board = board
 
    return board
 
 
# Helper/Auxiliary Functions
 
def goal_test(state):
    """
    Checks to see if current state is a goal state
    :param state: State of the board
    :type state: State
    :return: True if a state is a goal state, False otherwise
    :rtype: Bool
    """
    if state.board.grid[4][1] == char_goal and state.board.grid[4][2] == char_goal:
        return True
    return False
 
 
def calc_h(state):
    """
    Calculates the heuristic value of the given state using the heuristic function: Manhattan Distance (distance of tile from curr position to goal)
    :param state: State of the board
    :type state: State
    :return: The heuristic value or distance the 2x2 tile is from the goal
    :rtype: int
    """
 
    coords = state.board.get_goaltile_pos()
    if coords is None: #should never trigger
        return float('inf')
    return abs(coords[0] - 1) + abs(coords[1] - 3)
 
 
def generate_successors(state):
    """
    Function which takes a state and returns a list of its successor states.
    :param state: State of the board
    :type state: State
    :return: List of successor states
    :rtype: List
    """
    # take board of the state, loop through every piece and check all possible moves, if a move is possible,
    # make a new state with that move made and add it to the list of successors, otherwise keep going
 
    directions = ["up", "down", "left", "right"]
    successors = []
    coord_x = 0
    coord_y = 0
 
    for piece in state.board.pieces:
        for direction in directions:
            #check each piece if it can move in any direction, if it can get its new coords
            valid, coord_x, coord_y = piece.move_piece(direction)
            if valid: #iff move valid only then we deepcopy (was deepcopying before checking movevalid -> way higher runtime)
                new_board = deepcopy(state.board)
                for new_piece in new_board.pieces:
                    if repr(new_piece) == repr(piece): #find old piece
                        #update coords
                        new_piece.coord_x = coord_x
                        new_piece.coord_y = coord_y
                #init new state with board that has moved piece
                new_state = State(new_board, 0, state.depth + 1, state)
                #calc f
                new_state.f = calc_h(new_state) + new_state.depth
                successors.append(new_state)
    #debugging
    if successors:
        for s in successors:
            s.board.reconstruct_grid()
 
    return successors
 
 
def generate_initial_state(board):
    """
    Function that takes in the initial board and generates an associated initial state
    :param board: Initial board
    :type board: Board
    :return: Initial state
    :rtype: State
    """
 
    state = State(board, 0, 0, None)
    state.f = calc_h(state)
    return state
 
 
def get_solution(state):
    """
    Function that given a goal state returns a list of states ordered from goal all the way to initial (a solution to the puzzle)
    Return None if given state not a goal state
    :param state: Goal state
    :type state: Stat
    :return: Ordered list of states (from init to goal) that describe a solution to the puzzle
    :rtype: List
    """
 
    # first check if it's even a goal state
    if not goal_test(state):
        return None
 
    solution = []
 
    while state is not None:
        solution.append(state)
        state = state.parent
 
    solution.reverse()
 
    return solution
 
 
def output_file(state, filename):
    """
    given a state (goal), output the path taken to outputfile or name filename
    :param state, filename: State of the board (goal), desired name of output file
    :type state, filename: State, string
    :return: Does not return, output file detailing init state to goal state generated
    :rtype: Void (output file)
    """
 
    solution = get_solution(state)
    with open(filename, "w") as f:
        for states in solution:
            for line in states.board.grid:
                f.write(''.join(line) + '\n')
            f.write('\n')
 
 
def solve_puzzle(state, algo, outputfile):
 
    """
    Run search algorithm on state (initial) and write to output file (seperate function)
    :param state, algo, outputfile: State of the board to be solved from, algorithm to use on puzzle, name of outputfile
    :type state, algo, outputfile: State, string, string
    :return: True if a state is a goal state, False otherwise
    :rtype: Bool
    """
    if algo == "dfs":
        print("Running DFS with Pruning")
        out = DFS_pruning(state)
    if algo == "astar":
        print("Running A* with Pruning")
        out = A_star(state)
    
    output_file(out, outputfile)
 
 
# ====================================================================================
# ============================    Search Algorithms    ===============================
# ====================================================================================
 
def DFS_pruning(state):
    """
    Function that takes in initial state and runs dfs with pruning looking for a goal state, returns the goal state or None if not found
    :param state: Initial state
    :type state: State
    :return: Goal state
    :rtype: State
    """
 
    frontier = [state]
    explored = set()
    
    while frontier:
        selected = frontier.pop()
        if selected in explored:
            continue
        
        explored.add(selected)
 
        if goal_test(selected):
            return selected
        for s in generate_successors(selected):
            if s not in explored:
                frontier.append(s)
            
    return None
 
 
def A_star(state):
    """
    Function that takes in initial state and runs A* looking for a goal state, returns the goal state or None if not found
    :param state: Initial state
    :type state: State
    :return: Goal state
    :rtype: State
    """
    frontier = [state]
    explored = set()
    while frontier:
        selected = heappop(frontier)
        if selected in explored:
            continue
        explored.add(selected)
        if goal_test(selected):
            return selected
        successors = generate_successors(selected)
        for s in successors:
            if s not in explored:
                heappush(frontier, s)
    return None
 
 
# ====================================================================================
# ==============================    Main Function   ==================================
# ====================================================================================
 
def hrd(inputfile, outputfile, algo):
 
    """
    Overarching function, takes in all arguments and solves puzzle, generates output file, called in main with input args given
    :param inputfile, outputfile, algo: input file name, output file name, algorithm name
    :type inputfile, outputfile, algo: string, string, string
    :return: generate an output file with the name specified in output file 
    :rtype: Void (outputfile)
    """
 
    board = read_from_file(inputfile)
    init_state = generate_initial_state(board)
 
    start_time = time.time()
    solve_puzzle(init_state, algo, outputfile)
    end_time = time.time()
 
    time_taken = end_time - start_time
    print(f"Execution time: {time_taken:.6f} seconds") 
 
 
 
# =====
# MAIN
# =====
 
if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="The input file that contains the puzzle."
    )
    parser.add_argument(
        "--outputfile",
        type=str,
        required=True,
        help="The output file that contains the solution."
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=['astar', 'dfs'],
        help="The searching algorithm."
    )
    args = parser.parse_args()
 
    hrd(args.inputfile, args.outputfile, args.algo)