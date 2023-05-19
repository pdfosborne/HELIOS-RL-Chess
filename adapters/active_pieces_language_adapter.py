from typing import Dict, List
import pandas as pd
import torch
from torch import Tensor

import chess
from chess import Board

# StateAdapter includes static methods for adapters
from adapters.adapter_abstract import StateAdapter 
from helios_rl.encoders.sentence_transformer_MiniLM_L6v2 import LanguageEncoder

class ActivePiecesLanguageAdapter(StateAdapter):
    _cached_state_idx: Dict[str, int] = dict()

    def __init__(self):
        self.encoder = LanguageEncoder()
        self.start_name_lookup: dict = {'1':{'a':"White Queen's Rook", 'b':"White Queen's Knight", 'c':"White Queen's Bishop", 'd':"White Queen", 
                                            'e':"White King", 'f':"White King's Bishop", 'g':"White King's Knight",'h':"White King's Rook"},
                                        '2':{'a':"White Queen Rook's Pawn", 'b':"White Queen Knight's Pawn", 'c':"White Queen Bishop's Pawn", 'd':"White Queen's Pawn", 
                                            'e':"White King's Pawn", 'f':"White King Bishop's Pawn", 'g':"White King Knight's Pawn",'h':"White King Rook's Pawn"},
                                        '8':{'a':"Black Queen's Rook", 'b':"Black Queen's Knight", 'c':"Black Queen's Bishop", 'd':"Black Queen", 
                                            'e':"Black King", 'f':"Black King's Bishop", 'g':"Black King's Knight",'h':"Black King's Rook"},
                                        '7':{'a':"Black Queen Rook's Pawn", 'b':"Black Queen Knight's Pawn", 'c':"Black Queen Bishop's Pawn", 'd':"Black Queen's Pawn", 
                                            'e':"Black King's Pawn", 'f':"Black King Bishop's Pawn", 'g':"Black King Knight's Pawn",'h':"Black King Rook's Pawn"}}
    
    def adapter(self, board_fen:str, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """ Use Language name for every ACTIVE piece name for current board position."""
        #board = chess.Board(board_fen) # not used in this adapter so not calling
        # Not perfect, if piece ended up back in starting position then it's deemed 'inactive'
        if len(episode_action_history)==0:
            state = ['No pieces currently active on board.']
            self.active_pieces: dict = {}
        else:
            last_move = episode_action_history[-1]                
            start = last_move[:2]
            end = last_move[2:] # --> e4
            if start[1] in self.start_name_lookup:
                if start[0] in self.start_name_lookup[start[1]]:
                    piece_name = self.start_name_lookup[start[1]][start[0]]
                    if piece_name not in self.active_pieces:
                        self.active_pieces[piece_name] = {}
        
            state:str = 'The active pieces on the board are: '
            for n,active_piece in enumerate(list(self.active_pieces.keys())):
                if n+1 < len(list(self.active_pieces.keys())):
                    state = state + active_piece + ', '
                else:
                    state = state + active_piece + '.'
            state = [state]
        # Encode to Tensor for agents
        if encode:
                state_encoded = self.encoder.encode(state=state)
        else:
            state_encoded = state

        if (indexed):
            state_indexed = list()
            for sent in state:
                if (sent not in ActivePiecesLanguageAdapter._cached_state_idx):
                    ActivePiecesLanguageAdapter._cached_state_idx[sent] = len(ActivePiecesLanguageAdapter._cached_state_idx)
                state_indexed.append(ActivePiecesLanguageAdapter._cached_state_idx[sent])

            state_encoded = torch.tensor(state_indexed)

        return state_encoded
    
    def sample():
        board = chess.Board(fen='rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2')
        legal_moves = ['g1h3', 'g1f3', 'g1e2', 'f1a6', 'f1b5', 'f1c4', 'f1d3', 
                       'f1e2', 'e1e2', 'd1h5', 'd1g4', 'd1f3', 'd1e2', 'b1c3', 
                       'b1a3', 'e4e5', 'h2h3', 'g2g3', 'f2f3', 'd2d3', 'c2c3', 
                       'b2b3', 'a2a3', 'h2h4', 'g2g4', 'f2f4', 'd2d4', 'c2c4', 'b2b4', 'a2a4']
        episode_action_history = ['e2e4', 'c7c5']
        adapter = ActivePiecesLanguageAdapter()
        state = adapter.adapter(board, legal_moves, [], encode=False)
        state = adapter.adapter(board, legal_moves, [episode_action_history[0]], encode=False)
        state = adapter.adapter(board, legal_moves, episode_action_history, encode=False)
        # ---
        adapter = ActivePiecesLanguageAdapter()
        state_encoded = adapter.adapter(board, legal_moves, [], encode=True)
        state_encoded = adapter.adapter(board, legal_moves, [episode_action_history[0]], encode=True)
        state_encoded = adapter.adapter(board, legal_moves, episode_action_history, encode=True)

        return state, state_encoded