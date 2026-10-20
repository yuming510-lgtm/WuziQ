# V4C 运行与恢复

## 执行顺序

```bash
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. python scripts/preflight_bcd_high_compute_v4c.py
PYTHONPATH=. python scripts/run_bcd_high_compute_v4c.py
PYTHONPATH=. python scripts/summarize_bcd_high_compute_v4c.py
```

运行器每局结束后先校验整局，再将 JSONL 结果追加并 `fsync`，之后原子更新 `progress.json`。对局 ID 是唯一键，重启时跳过已完成 ID；未完成对局不会进入正式 JSONL。

## V4C Drive 恢复

恢复 ZIP 仅包含 `config.json`、`progress.json`、`9x9_games.jsonl` 和 `code_version.txt`，并为每个文件、完成 ID 集及归档本身记录 SHA-256。`restore_bcd_high_compute_v4c_snapshot.py` 会拒绝非空目标目录，检查归档摘要、重新校验棋谱、比对已完成 ID 和代码版本。

首次正式对局前先运行一次 `--max-games 0` 生成空进度恢复包，上传到 Drive 并实际下载、验 SHA、恢复到空目录。随后至少每 24 小时用最新 `progress.json` 与完整棋谱创建新快照：新文件上传后下载、校验并恢复成功，再删除上一份 V4C 文件。可在每若干局额外更新快照，以缩短环境重启后的重赛范围。所有操作只针对 `WuziQ V4C recovery latest.zip`，不得触碰 V1–V4B 快照或其他文件。

云环境丢失时，从 V4C 专用 Drive 文件下载 ZIP 到新工作目录，先以 SHA-256 和空目录恢复校验完成的 game ID，再恢复到 `research/v4c/results` 并重新运行 CLI。若没有有效 Drive 快照但本地棋谱在，则运行器直接按本地唯一 ID 续跑。

本实验不保存单局未结束时的 MCTS 内存树。快照边界是完整对局：若进程在一局中断，该局尚未落盘，会以固定配置/种子重新开始；所有已完成局保留且不会重复计数。

## 比赛约束

配置中锁定 B=1 秒、C=0.2 秒、D=50 秒，三个配对、三种开局以及两种换色。V4C 的 `engine_sha256` 固定 V4 的 B/C 搜索和 V4B 的 MCTS 源码。修改预算以外的 D 选点、root tactics、UCT、rollout 或树复用需要新版本研究，不能静默改入本实验。
