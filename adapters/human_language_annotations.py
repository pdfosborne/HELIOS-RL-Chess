from typing import Dict, List
from functools import lru_cache
import pandas as pd
import torch
from torch import Tensor
import json
import chess
from chess import Board

from adapters.adapter_abstract import StateAdapter
from helios_rl.encoders.sentence_transformer_MiniLM_L6v2 import LanguageEncoder
from adapters.board_adapter import BoardAdapter

class HumanAnnotationsAdapter(StateAdapter):
    _cached_state_idx: Dict[str, int] = dict()

    def __init__(self):
        # Import annotation info
        with open("./language_info/commentary.json") as annotated_games:
            self.annotations = {" ".join(fen.split()[:3]): annot for fen, annot in json.load(annotated_games).items()}    

        self.encoder = LanguageEncoder()
        self.board_to_adapter = BoardAdapter()
        self.prior_state = 'None'
        
    def adapter(self, board_fen:str, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """ Use NL name for every piece name for current board position """

        board = " ".join(board_fen.split()[:3])
        if board in self.annotations:
            annotation = self.annotations[board]
            # If multiple entries, select longest.
            if len(annotation) > 1:
                max_len = len(annotation[0])
                state = annotation[0]
                for label in annotation[1:]:
                    if len(label)>max_len:
                        state = label   
            else:
                state = annotation[0]
            state = state.split(".")
            state = [s for s in state if len(s) >=3]
            
            # state = [f"{piece['piece_des_name']} at {piece['board_pos']}" 
            #   for piece in board_CURRENT_Lang if (piece["piece_des_name"] != ".")]
            # Encode to Tensor for agents
            if encode:
                state_encoded = self.encoder.encode(state)
            else:
                state_encoded = state
        else:
            # backup if board not in lookup
            state = [self.prior_state + " progressing"]
            self.prior_state = state[-1]
            if encode:
                state_encoded = self.encoder.encode(state)
            else:
                state_encoded = state

        if (indexed):
            state_indexed = list()
            for sent in state:
                if (sent not in HumanAnnotationsAdapter._cached_state_idx):
                    HumanAnnotationsAdapter._cached_state_idx[sent] = len(HumanAnnotationsAdapter._cached_state_idx)
                state_indexed.append(HumanAnnotationsAdapter._cached_state_idx[sent])

            state_encoded = torch.tensor(state_indexed)
        
        return state_encoded
    
    def sample():
        board = chess.Board(fen='rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2')
        legal_moves = ['g1h3', 'g1f3', 'g1e2', 'f1a6', 'f1b5', 'f1c4', 'f1d3', 
                       'f1e2', 'e1e2', 'd1h5', 'd1g4', 'd1f3', 'd1e2', 'b1c3', 
                       'b1a3', 'e4e5', 'h2h3', 'g2g3', 'f2f3', 'd2d3', 'c2c3', 
                       'b2b3', 'a2a3', 'h2h4', 'g2g4', 'f2f4', 'd2d4', 'c2c4', 'b2b4', 'a2a4']
        episode_action_history = ['e2e4', 'c7c5']
        adapter = HumanAnnotationsAdapter()
        state = adapter.adapter(board, legal_moves, episode_action_history, encode=False)
        state_encoded = adapter.adapter(board, legal_moves, episode_action_history, encode=True)
        return state, state_encoded