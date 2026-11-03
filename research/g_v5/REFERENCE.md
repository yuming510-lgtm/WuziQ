# Independent AlphaZero reference

The V5 reference is `junxiaosong/AlphaZero_Gomoku`, fetched from
`https://github.com/junxiaosong/AlphaZero_Gomoku` at upstream commit
`0b4917ccd84ea2771706701015eb26182c1159f4` (2026-11-03). The upstream
repository carries the MIT license; its original license and README are kept
under `third_party/AlphaZero_Gomoku/`.

The vendored `game.py`, `mcts_alphaZero.py`, `mcts_pure.py`, and the original
PyTorch policy/value network are the reference code. The project runner imports
the reference board, network, `MCTS`, and `MCTSPlayer`. The self-play loop and
training update follow upstream `Game.start_self_play` and `TrainPipeline`
semantics while adding atomic checkpoints, unique game IDs, and telemetry.
`policy_value_net_pytorch.py` is not modified. A compatibility wrapper replaces
the obsolete `loss.data[0]` return with `.item()` and runs inference under
`torch.inference_mode()`; optimizer, loss, learning-rate adaptation, and search
math remain the upstream behavior.

The README recommends 6x6/connect-four as a learning smoke test and describes
400 playouts, roughly 500–1000 self-play games, and 8x symmetry augmentation.
These are upstream guidance, not results independently verified by this study.
The checked-in historical model files and GIF are not used as training data or
initialization. All V5 models start from seeded random weights.

This project uses the upstream native 4-plane, 32/64/128 convolutional model.
For comparison, WuziQ G0 is its frozen 32-channel, two-residual-block model.
Both have roughly comparable small-model scale, but architecture and input
encoding are not identical; the model-capacity difference remains a limitation.
