from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import pandas as pd
import json
from torch import Tensor
from functools import lru_cache

import chess
from chess import Board, SQUARES_180

# Import piece name lookup table
with open("./language_info/piece_names.json") as piece_name_map_file:
    PIECE_NAME_LOOKUP: Dict[str, Dict[str, str]] = json.load(piece_name_map_file)
    
PROMO_CHOICE_MAP = {
"r": "rook",
"n": "knight",
"b": "bishop",
"q": "queen",
"k": "king"
}
    
# Import language move logic
LOGIC_DF = pd.read_csv('./language_info/piece_logics.csv')
    
class Adapter(ABC):
    @abstractmethod
    def adapter(self, *args, **kwargs) -> List[str]:
        pass

class StateAdapter(Adapter):  
    
    @staticmethod
    def chess_object_lst() -> List[str]:
        chess_pieces = ['K','Q','R','B','N','P', 
                        'k','q','r','b','n','p']
        return chess_pieces
    
    @staticmethod
    def chess_poss_actions_lst() -> List[str]:
        cols = ['a','b','c','d','e','f','g','h']
        rows = ['1','2','3','4','5','6','7','8']
        all_possible_actions = []
        for c_1 in cols:
            for r_1 in rows:
                start = c_1+r_1
                for c_2 in cols:
                    for r_2 in rows:
                        end = c_2+r_2
                        all_possible_actions.append(start+end)
        # Pawn Promotions
        White_promotion_codes = ['R','N','B','Q','K']
        Black_promotion_codes = ['r','n','b','q','k']
        for c_1 in cols:
            for promo_code in White_promotion_codes:
                start = c_1+'7'
                end = c_1+'8'
                all_possible_actions.append(start+end+promo_code)            
            for promo_code in Black_promotion_codes:
                start = c_1+'2'
                end = c_1+'1'
                all_possible_actions.append(start+end+promo_code)
        # Pawns can go diagonal one square as well for pawn promo
        for c in range(0,len(cols)):
            if c==0:
                c_1 = cols[c]
                c_2 = cols[c+1]
                for promo_code in White_promotion_codes:
                    start = c_1+'7'
                    end = c_2+'8'
                    all_possible_actions.append(start+end+promo_code)
                for promo_code in Black_promotion_codes:
                    start = c_1+'2'
                    end = c_2+'1'
                    all_possible_actions.append(start+end+promo_code)
            elif c==7:
                c_1 = cols[c]
                c_2 = cols[c-1]
                for promo_code in White_promotion_codes:
                    start = c_1+'7'
                    end = c_2+'8'
                    all_possible_actions.append(start+end+promo_code)
                for promo_code in Black_promotion_codes:
                    start = c_1+'2'
                    end = c_2+'1'
                    all_possible_actions.append(start+end+promo_code)
            else:
                c_1 = cols[c]
                c_2 = cols[c-1]
                for promo_code in White_promotion_codes:
                    start = c_1+'7'
                    end = c_2+'8'
                    all_possible_actions.append(start+end+promo_code)
                for promo_code in Black_promotion_codes:
                    start = c_1+'2'
                    end = c_2+'1'
                    all_possible_actions.append(start+end+promo_code)
                c_1 = cols[c]
                c_2 = cols[c+1]
                for promo_code in White_promotion_codes:
                    start = c_1+'7'
                    end = c_2+'8'
                    all_possible_actions.append(start+end+promo_code)
                for promo_code in Black_promotion_codes:
                    start = c_1+'2'
                    end = c_2+'1'
                    all_possible_actions.append(start+end+promo_code)
        return all_possible_actions
    
      
    @staticmethod
    def compact_lst(board: Board) -> List[str]:
        builder = ["."] * len(SQUARES_180)
        for i, square in enumerate(SQUARES_180):
            piece = board.piece_at(square)

            if piece:
                builder[i] = piece.symbol()

        return builder
    
    @staticmethod
    def int_to_en(num):
        """Given an int32 number, print it in English."""
        d = { 0 : 'zero', 1 : 'one', 2 : 'two', 3 : 'three', 4 : 'four', 5 : 'five',
            6 : 'six', 7 : 'seven', 8 : 'eight', 9 : 'nine', 10 : 'ten',
            11 : 'eleven', 12 : 'twelve', 13 : 'thirteen', 14 : 'fourteen',
            15 : 'fifteen', 16 : 'sixteen', 17 : 'seventeen', 18 : 'eighteen',
            19 : 'nineteen', 20 : 'twenty',
            30 : 'thirty', 40 : 'forty', 50 : 'fifty', 60 : 'sixty',
            70 : 'seventy', 80 : 'eighty', 90 : 'ninety' }
        k = 1000
        m = k * 1000
        b = m * 1000
        t = b * 1000

        assert(0 <= num)

        if (num < 20):
            return d[num]

        if (num < 100):
            if num % 10 == 0: return d[num]
            else: return d[num // 10 * 10] + '-' + d[num % 10]

        if (num < k):
            if num % 100 == 0: return d[num // 100] + ' hundred'
            else: return d[num // 100] + ' hundred and ' + StateAdapter.int_to_en(num % 100)

        if (num < m):
            if num % k == 0: return StateAdapter.int_to_en(num // k) + ' thousand'
            else: return StateAdapter.int_to_en(num // k) + ' thousand, ' + StateAdapter.int_to_en(num % k)

        if (num < b):
            if (num % m) == 0: return StateAdapter.int_to_en(num // m) + ' million'
            else: return StateAdapter.int_to_en(num // m) + ' million, ' + StateAdapter.int_to_en(num % m)

        if (num < t):
            if (num % b) == 0: return StateAdapter.int_to_en(num // b) + ' billion'
            else: return StateAdapter.int_to_en(num // b) + ' billion, ' + StateAdapter.int_to_en(num % b)

        if (num % t == 0): return StateAdapter.int_to_en(num // t) + ' trillion'
        else: return StateAdapter.int_to_en(num // t) + ' trillion, ' + StateAdapter.int_to_en(num % t)

        raise AssertionError('num is too large: %s' % str(num))

    @staticmethod
    @lru_cache(maxsize=10000)
    def board_to_lang(board_fen: str):
        """ Output board us as a 2-d DataFrame with each board position and 
        the associated descriptive chess piece where . is still used to denote empty spaces. """
        # Board from engine needs to be flipped for White's POV
        board_flip = Board(board_fen)
        board_flip.apply_transform(chess.flip_vertical)
        # Transform into 1-D list
        board_lst = StateAdapter.compact_lst(board_flip)
        # Connect piece to grid location, SQUARE_NAMES defines 2-d position (e.g. e2) in a single list
        square_names_lst: List[str] = chess.SQUARE_NAMES
        # Rename each piece to simple naming convention (e.g. 'White King')
        board_df_src: List[Dict[str, str]] = list()
        for p in range(0, len(board_lst)):
            piece_des_name = 'init'
            piece_id = board_lst[p]
            board_pos = square_names_lst[p]
            # Extract piece descriptive name from lookup
            piece_des_name = PIECE_NAME_LOOKUP["piece_names"][piece_id]
            # Error handling if piece name is not overridden
            if (piece_des_name == 'init'):
                print("ERROR: board_to_lang_df function not mapping all pieces to names")
                print(piece_id)
            else:
                if piece_des_name != '.':
                    row = {"board_pos": board_pos, "player_name":piece_des_name.split(" ")[0] , "piece_id": piece_id, "piece_des_name": piece_des_name.split(" ")[1]}
                else:
                    row = {"board_pos": board_pos, "player_name":'.' , "piece_id":'.', "piece_des_name": '.'}
                board_df_src.append(row)
        return board_df_src

    @staticmethod
    def board_pos2piece_nm(board_fen:str, start_pos:str):
        piece_nm = 'init'
        # Find piece name based on current board configuration extracted in Language from board_to_lang
        board_current_lang = StateAdapter.board_to_lang(board_fen)
        for piece in reversed(board_current_lang):
            if (piece["board_pos"] == start_pos):
                if piece['player_name']=='.':
                    print("ERROR: Player Name not found for start pos - ", start_pos)
                    print(" ")
                    print(board_fen)
                piece_nm = piece["player_name"] + ' ' + piece["piece_des_name"]
                break
        return piece_nm

    @staticmethod
    def uci_to_lang_action(move_uci: str, board_fen: str):
        start_pos = move_uci[0:2]
        end_pos = move_uci[2:4]
        piece_nm = StateAdapter.board_pos2piece_nm(board_fen, start_pos)
        # Create Language based action based on piece name and start -> end grid position
        # - Pawn promo
        if (piece_nm != ".") and (piece_nm[6:]=='Pawn') and ((end_pos[1]=='8') or (end_pos[1]=='1')):
            promo_choice = move_uci[4].lower()
            promotion_piece = PROMO_CHOICE_MAP[promo_choice]
            # Pawn promo with capture        
            if start_pos[0] != end_pos[0]:
                lang_action = str(piece_nm) + ' at ' + str(start_pos) + ' captures a piece on ' + str(end_pos) + ' and is promoted to a ' + str(promotion_piece) 
            # Standard pawn promotion
            else: 
                lang_action = str(piece_nm) + ' at ' + str(start_pos) + ' promoted to a ' + str(promotion_piece) 
        # - Other unknown moves, e.g. castling
        elif piece_nm == 'init':
            print("Error: Invalid move_uci, no piece name can be found")
            print("Input uci:", move_uci)
            print(Board(board_fen))
            lang_action = "ERROR"
        # - Most moves
        else:
            lang_action = str(piece_nm) + " from " + str(start_pos) + " to " + str(end_pos)
        return lang_action
        

    @staticmethod
    def move_logics(player_nm: str, start_i: str, end_i: str, start_j: int, end_j: int, LANG_action) -> Tuple[str, int]:
        # White move logics
        if (player_nm == 'White'):
            if (start_j < end_j) & (start_i < end_i):
                move_dir = 'forwards and right'
            elif (start_j < end_j) & (start_i > end_i):
                move_dir = 'forwards and left'
            elif (start_j < end_j):
                move_dir = 'forwards'
            elif (start_j > end_j) & (start_i < end_i):
                move_dir = 'backwards and right'
            elif (start_j > end_j) & (start_i > end_i):
                move_dir = 'backwards and left'
            elif (start_j > end_j):
                move_dir = 'backwards'
            elif (start_i < end_i):
                move_dir = 'right'
            elif (start_i > end_i):
                move_dir = 'left'
            else:
                print("Error: invalid move direction")
                print(LANG_action)

        # Black move logics
        elif (player_nm == 'Black'):
            if (start_j > end_j) & (start_i > end_i):
                move_dir = 'forwards and right'
            elif (start_j > end_j) & (start_i < end_i):
                move_dir = 'forwards and left'
            elif (start_j > end_j):
                move_dir = 'forwards'
            elif (start_j < end_j) & (start_i > end_i):
                move_dir = 'backwards and right'
            elif (start_j < end_j) & (start_i < end_i):
                move_dir = 'backwards and left'
            elif (start_j < end_j):
                move_dir = 'backwards'
            elif (start_i > end_i):
                move_dir = 'right'
            elif (start_i < end_i):
                move_dir = 'left'
            else:
                print("Error: invalid move direction")
                print(LANG_action)
        else:
            print("Error: invalid player name")
            print(LANG_action)

        move_dis = abs(end_j - start_j) if not(move_dir in ["left", "right"]) else abs(ord(end_i) - ord(start_i))
        return move_dir, move_dis
        
    @staticmethod
    def action_to_lang(LANG_action: str, board_fen):
        LANG_action_split = LANG_action.split(" ")
        player_nm = LANG_action_split[0]
        piece_nm = LANG_action_split[1]
        start = LANG_action_split[3]
        start_i = start[0]
        start_j = int(start[1])
        # Pawn doesn't get changed
        if 'promoted' in LANG_action:
            LANG_action_description = LANG_action
        else:
            end = LANG_action_split[5]
            end_i = end[0]
            end_j = int(end[1])
            # Checks to see if final location matches an opponent's piece
            board_PRIOR_Lang = StateAdapter.board_to_lang(board_fen)
            end_piece = [piece for piece in board_PRIOR_Lang if (end in piece["board_pos"])][0]["piece_des_name"]
            captured_piece = end_piece.lower() if (end_piece != ".") else ""

            move_dir, move_dis = StateAdapter.move_logics(player_nm, start_i, end_i, start_j, end_j, LANG_action)
        
            piece_logic: Dict[Tuple[str], str] = {(r["Player"], r["Piece"], r["Move_dir"], r["Move_type"]): r["Language"] 
                                                    for r in LOGIC_DF.to_records()}
            if piece_nm == 'Pawn':
                if (move_dir == 'forwards'):
                    language = piece_logic[(player_nm, piece_nm, move_dir, "moves")]
                elif move_dir.split(' ')[2] in ['right', 'left']:
                    language = piece_logic[(player_nm, piece_nm, move_dir, "captures piece [N] by moving diagonally")]
                else:
                    print("ERROR")
            else:
                if captured_piece != '':
                    language = piece_logic.get((player_nm, piece_nm, move_dir, "captures piece [N] by moving"), None)
                    if (not language):
                        language = piece_logic[(player_nm, piece_nm, move_dir, "captures piece [N] by moving diagonally")]
                else:
                    language = piece_logic.get((player_nm, piece_nm, move_dir, "moves"), None)
                    if (not language):
                        language = piece_logic[(player_nm, piece_nm, move_dir, "moves diagonally")]
                                        
            desc = language.replace('{ij}', start) # Replaces string with piece start pos
            if 'captures' in desc:
                LANG_action_description = desc.replace('[N]', captured_piece)
            else:
                LANG_action_description = desc 
            if piece_nm == 'knight':
                desc_split = LANG_action_description.split('|')
                desc_split_sub = desc_split[2].split('*')
                LANG_action_description = desc_split[0] + str(move_dis) + desc_split_sub[0] + str(abs(end_i-start_i) )
            else:
                desc_split = LANG_action_description.split('|')
                LANG_action_description = desc_split[0] + str(move_dis) + desc_split[2]
        return LANG_action_description
    
    def adapter(self, board_fen:str, legal_actions:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """All adapters must output Tensor, use pre-built Encoders in the Helios package to tranform states to this form."""
        pass

    def sample():
        "Return a sample of the state adapted form (non-encoded)."
        pass
