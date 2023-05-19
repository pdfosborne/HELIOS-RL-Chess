from tqdm import tqdm
import time
# ------ Imports -----------------------------------------
from environment.engine import Engine
# Agent Setup
from helios_rl.environment_setup.imports import ImportHelper
# Evaluation standards
from helios_rl.environment_setup.results_table import ResultsTable
from helios_rl.environment_setup.helios_info import HeliosInfo
# ------ Chess specific imports --------------------------
import numpy as np
import chess
# ------ Opponent Agents -----------------------------------------
# Chess uniquely requires an opponent player for the probabilistic environment
# We utilize pre defined ones available in HELIOS for this but can be setup independently as part of the env
from helios_rl.agents.random_agent import RandomAgent
from environment.opponent_agents.sampled_agent import SampledAgent
OPPONENT_AGENT_TYPES = {
    "Random": RandomAgent,
    "Sampled": SampledAgent
}
OPPONENT_AGENT_PARAMETERS = {
    "Random":{},
    "Sampled":{}
}
# ------ State Adapters -----------------------------------------
# 1. Default Forms
from adapters.board_adapter import BoardAdapter
from adapters.board_pieces_adapter import BoardPiecesAdapter

STATE_ADAPTER_TYPES = {
    "Engine": BoardAdapter,
    "Board_as_pieces": BoardPiecesAdapter
}
# --------------------------------------------------------

class Environment:

    def __init__(self, local_setup_info: dict):
        # --- INIT env from engine
        self.env = Engine()
        self.start_obs = self.env.reset()
        # ---
        # --- PRESET HELIOS INFO
        # Agent
        Imports = ImportHelper(local_setup_info)
        self.agent, self.agent_type, self.agent_name, self.agent_state_adapter = Imports.agent_info(STATE_ADAPTER_TYPES)
        self.num_train_episodes, self.num_test_episodes, self.training_action_cap, self.testing_action_cap, self.reward_signal = Imports.parameter_info()  
        # Training or testing phase flag
        self.train = Imports.training_flag()
        # --- HELIOS
        self.live_env, self.observed_states, self.experience_sampling = Imports.live_env_flag()
        # Results formatting
        self.results = ResultsTable(local_setup_info)
        # HELIOS input function
        # - We only want to init trackers on first batch otherwise it resets knowledge
        self.helios = HeliosInfo(self.observed_states, self.experience_sampling)
        # Env start position for instr input
        # Enable sub-goals
        if (local_setup_info['sub_goal'] is not None) & (local_setup_info['sub_goal']!=["None"]) & (local_setup_info['sub_goal']!="None"):
            self.sub_goal:list = local_setup_info['sub_goal']
        else:
            self.sub_goal:list = None
            
        # --- CHESS SPECIFIC
        # Opponent agent is unique to Chess as part of the Probabilistic environment
        # But for ease we utilize the agent functions within HELIOS for the opponent
        train_opponent = local_setup_info['training_opponent_agent']
        training_opponent_parameters = OPPONENT_AGENT_PARAMETERS[train_opponent]
        self.training_opponent = OPPONENT_AGENT_TYPES[train_opponent](**training_opponent_parameters) 
        self.training_opponent_name = str(train_opponent) + '_' + str(training_opponent_parameters)
        # Different testing opponent to setup generalisability measurements when testing
        test_opponent = local_setup_info['testing_opponent_agent']
        testing_opponent_parameters = OPPONENT_AGENT_PARAMETERS[test_opponent]
        self.testing_opponent = OPPONENT_AGENT_TYPES[test_opponent](**testing_opponent_parameters) 
        self.testing_opponent_name = str(test_opponent) + '_' + str(testing_opponent_parameters)
        # ---

    @staticmethod
    def goal_reached(current_board_fen, sub_goal_board, engine_terminated, action_num, action_cap):
        current_board = chess.Board(current_board_fen)
        # Engine terminal state reached
        if engine_terminated is True:
            game_over = True
        # Action cap reached
        elif action_num == action_cap:
            game_over = True
        # Sub-goal reached
        elif 'first_capture' in sub_goal_board:
            if np.sum([current_board.piece_type_at(sq) for sq in chess.SQUARES if current_board.piece_type_at(sq) is not None])<74:
                game_over = True
            else:
                game_over = False
        elif (sub_goal_board != "None")&(str(current_board.fen()) in sub_goal_board):
            game_over = True
        else:
            game_over = False
        return game_over
        
    @staticmethod
    def reward(reward_signal, current_board_fen, player_turn, action_num, action_cap, game_over) -> float:
        current_board = chess.Board(current_board_fen)
        if game_over is True:
            game_result = current_board.result() # If game not ended == '*'
            # Action limit reached so draw (stalemate)
            if (game_result == '*')&(action_num == action_cap):
                r = reward_signal[1]
            # Game over is true but match is not ended and not reached action limit so sub-goal found
            elif game_result == '*':
                # If reward signal is based on first capture or if opponent makes action to reach sub-goal
                if player_turn == 'white':
                    r = reward_signal[0]
                else:
                    r = reward_signal[0]*-1
            # Match ended
            else:     
                if game_result == '1/2-1/2':
                    r = reward_signal[1]
                elif game_result == '1-0':
                    r = reward_signal[0]
                elif game_result == '0-1':                
                    r = reward_signal[0]*-1
                else:
                    print("ERROR: Unknown game result for reward function.")
        # Else immediate reward for any action
        else:
            r = reward_signal[2]     
        return r

    def episode_loop(self):
        # Mode selection (already initialized)
        if self.train:
            BLACK_AGENT = self.training_opponent
            black_player_name = self.training_opponent_name
            number_episodes = self.num_train_episodes
            action_cap = self.training_action_cap
        else:
            BLACK_AGENT = self.testing_opponent
            black_player_name = self.training_opponent_name
            number_episodes = self.num_test_episodes
            action_cap = self.testing_action_cap

        for episode in tqdm(range(0, number_episodes)):
            action_history = []
            # ---
            # Start observation is used instead of .reset() fn so that this can be overriden for repeat analysis from the same start pos
            obs = self.env.reset() # In this case we can hard reset the env because chess has a fixed start
            legal_moves = self.env.legal_move_generator(obs)
            state = self.agent_state_adapter.adapter(board_fen=obs, legal_moves=legal_moves, episode_action_history=action_history, encode=True)
            # ---
            start_time = time.time()
            episode_reward:int = 0
            for action in range(0,self.training_action_cap):
                if self.live_env:
                    # Agent takes action
                    legal_moves = self.env.legal_move_generator(obs)
                    agent_action = self.agent.policy(state, legal_moves)
                    action_history.append(agent_action)
                    # Push move into board engine
                    next_obs, reward, engine_terminated = self.env.step(state=obs, action=agent_action)
                    legal_moves = self.env.legal_move_generator(next_obs) 
                    next_state = self.agent_state_adapter.adapter(board_fen=next_obs, legal_moves=legal_moves, episode_action_history=action_history, encode=True)
                    # ---
                    # Game over check
                    terminated = Environment.goal_reached(current_board_fen=next_obs, 
                                        sub_goal_board=self.sub_goal, 
                                        engine_terminated=engine_terminated, 
                                        action_num=action, action_cap=action_cap)
                    # Reward signal function
                    reward = Environment.reward(self.reward_signal, next_obs, 'white', action, action_cap, terminated)
                    # ---
                    
                    # HELIOS trackers    
                    self.helios.observed_state_tracker(engine_observation=next_obs,
                                                        language_state=self.agent_state_adapter.adapter(board_fen=next_obs, legal_moves=legal_moves, episode_action_history=action_history, encode=False))
                    
                    # MUST COME BEFORE SUB-GOAL CHECK OR 'TERMINAL STATES' WILL BE FALSE
                    self.helios.experience_sampling_add(state, agent_action, next_state, reward, terminated)
                    # CHESS: NOT NEEDED HERE BECAUSE ITS PERFORMED IN GOAL_REACHED CHECK
                    # # Trigger end on sub-goal if defined
                    # if self.sub_goal:
                    #     if (type(self.sub_goal)==type(''))|(type(self.sub_goal)==type(0)):
                    #         if next_obs == self.sub_goal:
                    #             reward = self.reward_signal[0]
                    #             terminated = True
                    #     elif (type(self.sub_goal)==type(list('')))|(type(self.sub_goal)==type(list(0))):    
                    #         if next_obs in self.sub_goal:
                    #             reward = self.reward_signal[0]
                    #             terminated = True         
                    #     else:
                    #         print("Sub-Goal Type ERROR: The input sub-goal type must be a str/int or list(str/int).")               
                else:
                    # Experience Sampling
                    legal_moves = self.helios.experience_sampling_legal_actions(state)
                    # Unknown state, have no experience to sample from so force break episode
                    if legal_moves == None:
                        break
                    
                    agent_action = self.agent.policy(state, legal_moves)
                    next_state, reward, terminated = self.helios.experience_sampling_step(state, agent_action)

                if self.train:
                    self.agent.learn(state, next_state, reward, agent_action)
                episode_reward+=reward
                if terminated:
                    break    
                # ---------------------------
                # Then Black turn (opponent)   
                if self.live_env:
                    obs = next_obs
                    #action_num+=1 # Don't increase action count for black players action
                    legal_moves = self.env.legal_move_generator(obs)
                    # ------------ ENGINE ACTION ------------
                    # Snapshot previous board/state/move before action gets made by agent for next adapter
                    black_action = BLACK_AGENT.policy(obs, legal_moves)
                    action_history.append(black_action)
                    # Push move into board engine
                    next_obs, reward, engine_terminated = self.env.step(state=obs, action=black_action)
                    legal_moves = self.env.legal_move_generator(obs)
                    # Game over check
                    terminated = Environment.goal_reached(current_board_fen=next_obs, 
                                        sub_goal_board=self.sub_goal, 
                                        engine_terminated=engine_terminated, 
                                        action_num=action, action_cap=action_cap)
                    # End episode
                    if terminated:
                        # Reward signal function
                        reward = Environment.reward(self.reward_signal, next_obs, 'black', action, action_cap, terminated)
                        episode_reward+=reward
                        # In the case the black player ends the game, update white's knowledge with their last move + new reward
                        self.agent.learn(state, next_state, reward, agent_action)
                        break
                    else:
                        state=next_state
            # If action limit reached
            if not terminated:
                reward = self.reward_signal[2]     
                
            end_time = time.time()
            agent_results = self.agent.q_result()
            if self.live_env:
                self.results.results_per_episode(self.agent_name, black_player_name, episode, action, episode_reward, (end_time-start_time), action_history, agent_results[0], agent_results[1]) 

        return self.results.results_table_format()
                    
