import json
# Engine used to obtain move scores
import chess.engine
from chess import Board


    
def player_data(data) -> dict:
    move_df = {}
    board: Board = chess.Board() 
    for i in data:
        # Used for dict index
        # -> allows easily search for possible moves by board
        move_name = i['u']

        # Used for dict data
        move_seq = i['ml'].split(" ")
        move_seq_prev = move_seq[:-1]
        move = move_seq[-1]
        human_play_count = 0 # TODO missing this info -> Merged from other source
        win_perc = i['wl']
        
        # Extract setup board code 
        # -> data does not include this
        # -> so we play chess engine and extract the boards from that
        board.reset()
        for move_uci in move_seq_prev:
            board.push_san(board.san(chess.Move.from_uci(move_uci)))
        req_board = str(board.fen())
        # Extract outcome board as well for consistency
        if move != '': # Catch error input ("id": 2985) with no move
            board.push_san(board.san(chess.Move.from_uci(move)))
            outcome_board = str(board.fen())
            
            if req_board not in move_df:
                move_df[req_board] = {}
                
            move_df[req_board][move] = {'prev_moves_uci': move_seq_prev, 
                                            'move_name': move_name,
                                            'outcome_board': outcome_board,
                                            'count': human_play_count,
                                            'win_perc': win_perc} 
    return(move_df)

def main():
    with open('./eco_book_stats.json', 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        # Print the type of data variable
        #print(" ")
        #print("Loading the known human player moves data... ")
    output = player_data(data)
    with open("./move_stats.json", "w") as output_file:
        json.dump(output, output_file, indent=2)

if __name__ == "__main__":
    main()
