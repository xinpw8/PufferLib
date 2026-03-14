"""Pure self-play training with fixed ChessSeven (3x3 conv) and proper parallelization."""
import sys, os, time, torch
sys.path.insert(0, '/home/spark-advantage/pufferlib-4.0')

nccl_path = '/home/spark-advantage/.venv/lib/python3.12/site-packages/nvidia/nccl/lib'
torch_path = '/home/spark-advantage/.venv/lib/python3.12/site-packages/torch/lib'
os.environ['LD_LIBRARY_PATH'] = f"{nccl_path}:{torch_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

from pufferlib.pufferl import PuffeRL, load_config, Logger

args = load_config('puffer_chess')

# Pure selfplay overrides
args['env']['selfplay'] = 1
args['env']['turn_gating'] = 1
args['env']['random_bot'] = 0
args['env']['stockfish_bot'] = 0
args['env']['chess_encoder'] = 1  # ChessSeven (fixed 3x3 conv), not ChessTwoEncoder
args['env']['fen_curric_pct'] = 0.0
args['env']['deepmind_fen_pct'] = 0.0
args['env']['reward_tutor_piece'] = 0.0
args['env']['reward_tutor_move'] = 0.0
args['env']['reward_tutor_wrong'] = 0.0
args['env']['tutor_only_mode'] = 0
args['env']['mate_curriculum'] = 0
args['env']['puzzle_drill_mode'] = 0

# Reward shaping - keep material/check/draw/syzygy signals
args['env']['reward_material'] = 0.05
args['env']['reward_check'] = 0.01
args['env']['reward_mate'] = 5.0
args['env']['reward_draw'] = -0.5
args['env']['reward_syzygy'] = 0.5
args['env']['reward_repetition'] = -0.3
args['env']['max_moves'] = 200

# Training config
args['train']['total_timesteps'] = 50_000_000_000
args['train']['reward_clip'] = 1.0
args['train']['checkpoint_interval'] = 25

# PARALLELIZATION - fixed from 4096/4/32 to 16384/8/128
args['vec']['total_agents'] = 16384
args['vec']['num_buffers'] = 8
args['vec']['num_threads'] = 128

train_cfg = dict(args['train'])
train_cfg['env_name'] = args['env_name']
train_cfg['env'] = args['env_name']
train_cfg['use_rnn'] = args['rnn_name'] is not None

logger = Logger(args)
print(f"Starting pure self-play training (FIXED)")
print(f"  Network: ChessSeven (3x3 conv, ~340K params)")
print(f"  Agents: {args['vec']['total_agents']} (was 4096)")
print(f"  Buffers: {args['vec']['num_buffers']} (was 4)")
print(f"  Threads: {args['vec']['num_threads']} (was 32)")
print(f"  Encoder: chess_encoder=1 (ChessSeven, obs=1082)")
print(f"  Target: {args['train']['total_timesteps']/1e9:.0f}B steps")
print(f"  Reward clip: [-{args['train']['reward_clip']}, {args['train']['reward_clip']}]")

pufferl = PuffeRL(train_cfg, args['vec'], args['env'], args['policy'], logger=logger, verbose=True)

start = time.time()
last_log = 0
try:
    while pufferl.global_step < args['train']['total_timesteps']:
        pufferl.evaluate()
        pufferl.train()

        now = time.time()
        if now - last_log > 60:
            elapsed = now - start
            sps = pufferl.global_step / elapsed if elapsed > 0 else 0
            pct = 100 * pufferl.global_step / args['train']['total_timesteps']
            print(f"[{pct:5.1f}%] epoch={pufferl.epoch} step={pufferl.global_step/1e6:.1f}M SPS={sps/1e3:.1f}K elapsed={elapsed/60:.0f}min")
            last_log = now
except KeyboardInterrupt:
    print("Training interrupted by user")
finally:
    pufferl.close()
    print(f"Training complete. Final epoch={pufferl.epoch}, steps={pufferl.global_step/1e6:.1f}M")
