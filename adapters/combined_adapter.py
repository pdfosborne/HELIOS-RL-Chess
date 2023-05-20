from typing import Dict, List
import pandas as pd
import torch
from torch import Tensor
import chess
from chess import Board

# StateAdapter includes static methods for adapters
from adapters.adapter_abstract import StateAdapter 
from helios_rl.encoders.sentence_transformer_MiniLM_L6v2 import LanguageEncoder

from adapters.board_to_language_adapter import BoardToLanguageAdapter
from adapters.active_pieces_language_adapter import ActivePiecesLanguageAdapter
from adapters.prior_actions_to_language_adapter import PriorActionsToLanguageAdapter
from adapters.poss_actions_to_language_adapter import PossibleActionsToLanguageAdapter

class CombinedAdapter(StateAdapter):
    def __init__(self):
        self.BoardtoLanguage = BoardToLanguageAdapter()
        self.ActivePiecesLanguage = ActivePiecesLanguageAdapter()
        self.PriorActionstoLanguage = PriorActionsToLanguageAdapter()
        self.PossibleActionsToLanguage = PossibleActionsToLanguageAdapter()
        self.encoder = LanguageEncoder()
    
    def adapter(self, board_fen:str, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """ Combines all other adapters into a single state description """
        board: Board = chess.Board(board_fen)
        board_lang = self.BoardtoLanguage.adapter(board, legal_moves, episode_action_history, encode=False, indexed=indexed)
        active_pieces_lang = self.ActivePiecesLanguage.adapter(board, legal_moves, episode_action_history, encode=False, indexed=indexed)
        prior_action_lang = self.PriorActionstoLanguage.adapter(board, legal_moves, episode_action_history, encode=False, indexed=indexed)
        poss_action_lang = self.PossibleActionsToLanguage.adapter(board, legal_moves, episode_action_history, encode=False, indexed=indexed)

        state = board_lang + active_pieces_lang + prior_action_lang + poss_action_lang
        state.remove('')

        if (not indexed):
            if encode:
                state_encoded = self.encoder.encode(state=state)
            else:
                state_encoded = state
        else:
            state_encoded = torch.tensor(state)

        return state_encoded
    
    def sample():
        board = chess.Board(fen='rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2')
        legal_moves = ['g1h3', 'g1f3', 'g1e2', 'f1a6', 'f1b5', 'f1c4', 'f1d3', 
                       'f1e2', 'e1e2', 'd1h5', 'd1g4', 'd1f3', 'd1e2', 'b1c3', 
                       'b1a3', 'e4e5', 'h2h3', 'g2g3', 'f2f3', 'd2d3', 'c2c3', 
                       'b2b3', 'a2a3', 'h2h4', 'g2g4', 'f2f4', 'd2d4', 'c2c4', 'b2b4', 'a2a4']
        episode_action_history = ['e2e4', 'c7c5']
        adapter = CombinedAdapter()
        state = adapter.adapter(board, legal_moves, [], encode=False)
        state = adapter.adapter(board, legal_moves, [episode_action_history[0]], encode=False)
        state = adapter.adapter(board, legal_moves, episode_action_history, encode=False)
        adapter = CombinedAdapter()
        state_encoded = adapter.adapter(board, legal_moves, [], encode=True)
        state_encoded = adapter.adapter(board, legal_moves, [episode_action_history[0]], encode=True)
        state_encoded = adapter.adapter(board, legal_moves, episode_action_history, encode=True)
        return state, state_encoded