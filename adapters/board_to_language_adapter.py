from typing import Dict, List
import pandas as pd
import torch
from torch import Tensor

import chess
from chess import Board

# StateAdapter includes static methods for adapters
from adapters.adapter_abstract import StateAdapter 
from helios_rl.encoders.sentence_transformer_MiniLM_L6v2 import LanguageEncoder

class BoardToLanguageAdapter(StateAdapter):
    _cached_state_idx: Dict[str, int] = dict()

    def __init__(self):
        self.encoder = LanguageEncoder()
    
    def adapter(self, board_fen:str, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """ Use Language name for every piece name for current board position """
        board_CURRENT_Lang = StateAdapter.board_to_lang(board_fen)
        # state = [f"{piece['piece_des_name']} at {piece['board_pos']}" 
        #          for piece in board_CURRENT_Lang if (piece["piece_des_name"] != ".")]
        
        # Convert raw board as language dict to occurance counter
        occ_dict: Dict = {}
        for piece in board_CURRENT_Lang:
            if piece['piece_des_name'] !='.':
                player_name = piece['player_name']
                piece_name = piece['piece_des_name']
                
                if player_name not in occ_dict:
                    occ_dict[player_name] = {}
                    
                if piece_name not in occ_dict[player_name]:
                    occ_dict[player_name][piece_name] = {'count':1}
                else:
                    count = occ_dict[player_name][piece_name]['count']
                    occ_dict[player_name][piece_name]['count'] = count+1
        # Covert numeric dict to a list of strings describing player positions
        state:List[str] = []
        for player_name in list(occ_dict.keys()):
            player_str = 'The ' + str(player_name) + ' player has '
            for piece_name in list(occ_dict[player_name].keys()):
                # 'four pawns, '
                piece_str = str(StateAdapter.int_to_en(occ_dict[player_name][piece_name]['count'])) + ' ' + str(piece_name) + ', '
                player_str += piece_str
            full_str = player_str + 'left on the board.'
            state.append(full_str)
                    
        # Encode to Tensor for agents
        if encode:
            state_encoded = self.encoder.encode(state=state)
        else:
            state_encoded = state

        if (indexed):
            state_indexed = list()
            for sent in state:
                if (sent not in BoardToLanguageAdapter._cached_state_idx):
                    BoardToLanguageAdapter._cached_state_idx[sent] = len(BoardToLanguageAdapter._cached_state_idx)
                state_indexed.append(BoardToLanguageAdapter._cached_state_idx[sent])

            state_encoded = torch.tensor(state_indexed)

        return state_encoded
    
    def sample():
        board = chess.Board(fen='rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2')
        legal_moves = ['g1h3', 'g1f3', 'g1e2', 'f1a6', 'f1b5', 'f1c4', 'f1d3', 
                       'f1e2', 'e1e2', 'd1h5', 'd1g4', 'd1f3', 'd1e2', 'b1c3', 
                       'b1a3', 'e4e5', 'h2h3', 'g2g3', 'f2f3', 'd2d3', 'c2c3', 
                       'b2b3', 'a2a3', 'h2h4', 'g2g4', 'f2f4', 'd2d4', 'c2c4', 'b2b4', 'a2a4']
        episode_action_history = ['e2e4', 'c7c5']
        adapter = BoardToLanguageAdapter()
        state = adapter.adapter(board, legal_moves, episode_action_history, encode=False)
        state_encoded = adapter.adapter(board, legal_moves, episode_action_history, encode=True)
        return state,state_encoded