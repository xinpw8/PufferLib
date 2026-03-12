"""Auto-evaluate chess checkpoints against Stockfish as they appear."""
import glob, os, sys, time, json
sys.path.insert(0, '/home/spark-advantage/pufferlib-3.0-chess')
sys.path.insert(0, '/home/spark-advantage/pufferlib-3.0-chess/scripts')
import torch
import chess, chess.engine
import numpy as np
from pufferlib.ocean import torch as ocean_torch
from pufferlib.models import LSTMWrapper
from pufferlib.ocean.chess.chess import Chess
from chess_stockfish_bridge import (
    build_observation, count_repetitions, legal_destinations_for_source,
    move_to_actions, STARTING_FEN, OBS_SIZE
)

STOCKFISH = '/home/spark-advantage/Stockfish/src/stockfish'
ELO = 1320
NGAMES = 20
RESULTS_FILE = '/home/spark-advantage/pufferlib-3.0-chess/eval_results.json'
DEVICE = 'cuda'

def load_model(ckpt_path):
    dummy_env = Chess(num_envs=1, selfplay=1)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    has_lstm = any('lstm' in k for k in ckpt.keys())
    has_spatial = any('spatial_blocks' in k for k in ckpt.keys())
    
    if has_spatial:
        block_indices = set()
        for k in ckpt.keys():
            if 'spatial_blocks' in k:
                block_indices.add(int(k.split('spatial_blocks.')[1].split('.')[0]))
        num_spatial = len(block_indices)
        proj_w = [k for k in ckpt.keys() if 'channel_proj.weight' in k][0]
        proj_dim = ckpt[proj_w].shape[0]
        policy = ocean_torch.ChessLight(dummy_env, proj_dim=proj_dim, num_spatial=num_spatial,
                                         hidden_size=256, embed_dim=32)
    elif any('square_embed' in k for k in ckpt.keys()):
        policy = ocean_torch.ChessSeven(dummy_env, square_dim=64, proj_dim=8, hidden_size=256, embed_dim=64)
    else:
        dummy_env.close()
        return None, None, None
    
    if has_lstm:
        hs = policy.hidden_size
        model = LSTMWrapper(dummy_env, policy, input_size=hs, hidden_size=hs)
        model.load_state_dict(ckpt)
    else:
        policy.load_state_dict(ckpt)
        model = policy
    
    model = model.to(DEVICE).eval()
    dummy_env.close()
    return model, has_lstm, f'CL-{proj_dim}x{num_spatial}' if has_spatial else 'CS'

def play_game(model, has_lstm, learner_color):
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    engine.configure({'UCI_LimitStrength': True, 'UCI_Elo': ELO})
    limit = chess.engine.Limit(time=0.03)
    
    board = chess.Board(STARTING_FEN)
    state = {'lstm_h': None, 'lstm_c': None} if has_lstm else None
    flip = 56 if learner_color == chess.BLACK else 0
    moves = 0
    
    while not board.is_game_over() and moves < 500:
        if board.turn == learner_color:
            legal_moves = list(board.legal_moves)
            if not legal_moves: break
            
            # Phase 0: pick piece
            rep = count_repetitions(board)
            obs0 = build_observation(board, learner_color, 0, None, legal_moves, [], rep)
            full0 = np.zeros(OBS_SIZE * 2, dtype=np.uint8)
            full0[:OBS_SIZE] = obs0
            obs_t = torch.from_numpy(full0).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                if has_lstm:
                    out = model.forward_eval(obs_t, state)
                else:
                    out = model(obs_t)
                logits = out[0][0]
            
            valid_srcs = {m.from_square ^ flip for m in legal_moves}
            sl = logits[:64].clone()
            for sq in range(64):
                if sq not in valid_srcs: sl[sq] = -1e8
            src_action = torch.argmax(sl).item()
            src_sq = src_action ^ flip
            
            # Phase 1: pick dest
            dests = legal_destinations_for_source(legal_moves, src_sq)
            if not dests:
                import random
                move = random.choice(legal_moves)
            else:
                obs1 = build_observation(board, learner_color, 1, src_sq, legal_moves, dests, rep)
                full1 = np.zeros(OBS_SIZE * 2, dtype=np.uint8)
                full1[:OBS_SIZE] = obs1
                obs_t1 = torch.from_numpy(full1).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    if has_lstm:
                        out1 = model.forward_eval(obs_t1, state)
                    else:
                        out1 = model(obs_t1)
                    logits1 = out1[0][0]
                
                valid_das = {}
                for m in dests:
                    _, da = move_to_actions(m, learner_color)
                    valid_das[da] = m
                
                dl = logits1[:96].clone()
                for a in range(96):
                    if a not in valid_das: dl[a] = -1e8
                da = torch.argmax(dl).item()
                move = valid_das.get(da, list(valid_das.values())[0])
            
            board.push(move)
        else:
            result = engine.play(board, limit)
            board.push(result.move)
        moves += 1
    
    engine.quit()
    outcome = board.outcome()
    if outcome is None or outcome.winner is None:
        return 'draw', moves
    elif outcome.winner == learner_color:
        return 'win', moves
    else:
        return 'loss', moves

def eval_checkpoint(ckpt_path, ngames=NGAMES):
    model, has_lstm, mname = load_model(ckpt_path)
    if model is None:
        return None
    
    w = d = l = 0
    for i in range(ngames):
        color = chess.WHITE if i % 2 == 0 else chess.BLACK
        result, moves = play_game(model, has_lstm, color)
        if result == 'win': w += 1
        elif result == 'draw': d += 1
        else: l += 1
        c = 'W' if color == chess.WHITE else 'B'
        print(f'  G{i+1:02d}/{ngames} [{c}] {result:4s} ({moves:3d}mv) | W:{w} D:{d} L:{l}')
    
    return {'wins': w, 'draws': d, 'losses': l, 'games': ngames, 
            'win_rate': w/ngames, 'score': (w+0.5*d)/ngames}

def main():
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    
    while True:
        checkpoints = sorted(glob.glob('experiments/puffer_chess_*/model_*.pt'))
        new_ckpts = [c for c in checkpoints if c not in results]
        
        if new_ckpts:
            for ckpt in new_ckpts:
                print(f'\n=== Evaluating {os.path.basename(ckpt)} vs SF {ELO} ===')
                r = eval_checkpoint(ckpt)
                if r:
                    results[ckpt] = r
                    print(f'Result: {r[wins]}W {r[draws]}D {r[losses]}L ({100*r[win_rate]:.0f}% WR)')
                    with open(RESULTS_FILE, 'w') as f:
                        json.dump(results, f, indent=2)
        else:
            print(f'[{time.strftime(%H:%M:%S)}] No new checkpoints. Waiting...')
        
        time.sleep(120)  # Check every 2 minutes

if __name__ == '__main__':
    main()
