import json
# Engine used to obtain move scores
import chess.engine
from chess import Board

# Opening JSON file
with open('./language_info/known_states_tree.json', 'r') as json_file:
    data = json.load(json_file)
    # Print the type of data variable
    print(" ")
    print("Type:", type(data))
    print("=================")
    
class MoveDataExtract():
    def __init__(self):
        self.move_df: dict = {}
        self.board: Board = chess.Board()
        
    def reach_down(branch: dict, n: int):
        """Reaches down a branch. Children is a sub-list not dict making this difficult."""
        branch_down = branch["children"][n][list(branch["children"][n].keys())[0]]
        return branch_down

    def data_extract(self, move_list, branch):
        """Extracts the move UCI and generates the data entry for the output dict."""
        # Extract and Log move SAN code
        move_uci = self.board.parse_san(branch["data"]["sanMove"]).uci()        
        self.move_df[str(self.board.fen())][move_uci] = {'prev_moves_uci': move_list, 
                                                    'totalGames': branch["data"]["totalGames"]} 
        # Play move with Chess engine to get next board
        self.board.push_san(self.board.san(chess.Move.from_uci(move_uci)))
        self.move_df[str(self.board.fen())] = {}

    def move_list_generator(branch, n):
        """Generates move list from string. Converts SAN to UCI codes by sampling over board re-play."""
        move_list = str(list(branch["children"][n].keys())).replace('[', '').replace(']', '').replace('(', '').replace(')', '').replace('"', '').replace("'", '').split(", ")
        move_list_uci = []
        board_temp = chess.Board()
        for move_san in move_list:
            move_uci = board_temp.parse_san(move_san).uci()
            move_list_uci.append(move_uci)
            board_temp.push_san(move_san)
        move_list = move_list_uci
        return move_list
    
    def inner_loop(self, branch):
        layer = 0
        while "children" in branch: 
            layer += 1
            for n in range(0,len(branch["children"])):
                move_list = MoveDataExtract.move_list_generator(branch, n)
                branch = MoveDataExtract.reach_down(branch, n)
                MoveDataExtract.data_extract(self, move_list, branch)
                
                MoveDataExtract.inner_loop(self, branch)
                
    def tree_to_dict(self):
        """Start reference is slightly different so called manually. Then call recursive inner loop to sweep through branches. """        
        self.move_df[str(self.board.fen())] = {}
        for n in range(0,len(data["start"]["children"])): 
            self.board.reset()
            branch = MoveDataExtract.reach_down(data["start"], n)
            move_list = MoveDataExtract.move_list_generator(branch, n)
            
            MoveDataExtract.data_extract(self, move_list, branch)
            MoveDataExtract.inner_loop(self, branch)
        return(self.move_df)
    
if __name__ == "__main__":
    test = MoveDataExtract()
    dict = test.tree_to_dict_old()
    print(dict[list(dict.keys())[0]])