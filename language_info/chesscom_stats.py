import pickle
import json
import chess
from typing import Tuple, List, Dict
from chess import Board
from treelib import Tree, Node


def conv_tree(stats_tree: Tree, board: Board = None) -> Dict[str, Dict[str, dict]]:
    if (not board):
        board = chess.Board()

    stats_map = {board.fen(): dict()}
    children: List[Tree] = [stats_tree.subtree(c.identifier) for c in stats_tree.children(stats_tree.root)]

    if (children):
        for child in children:
            data = stats_tree[child.root].data
            next_b = board.copy()
            try:
                move_uci = board.uci(next_b.push_san(data["sanMove"]))
            except ValueError:
                print(child.root)
                continue
            stats_map[board.fen()][move_uci] = {
                "prev_moves_uci": [move.uci() for move in board.move_stack],
                "whiteWon": data["whiteWon"],
                "blackWon": data["blackWon"],
                "draw": data["draw"],
                "totalGames": data["totalGames"]
            }
            stats_map.update(conv_tree(child, next_b))
    
    return stats_map

def main():
    with open("known_states_tree.pickle", "rb") as known_states_file:
        known_states_tree = pickle.load(known_states_file)

    stats_map = conv_tree(known_states_tree)
    with open("stats_map.json", "w") as output_file:
        json.dump(stats_map, output_file, indent=2)

if __name__ == "__main__":
    main()
