# WuziQ G V4 恢复入口

Drive 专用目录：`WuziQ G V4 Recovery`（目录 ID `18cKYv8LqNpysonwLJyJdhVY6Ucu0yQos`）。只轮换此目录中的 V4 快照；V1–V3 等历史目录不动。

1. 从该目录下载最新 `g_v4_recovery_*.zip` 到本仓库分支对应工作树。
2. 先验证 SHA-256，再以空目录执行：

   ```sh
   python scripts/restore_g_v4_snapshot.py --archive /path/to/g_v4_recovery_latest.zip --destination /tmp/g_v4_restore_check
   ```

3. 将恢复树中的 `research/g_v4/` 与 `research/g_v3/checkpoints/G0/seed-510901/node_20000.pt`、`research/g_v3/results/selfplay/G0-seed-510901.jsonl`、`research/f_v2/checkpoints/best_f2.pt` 按原相对路径还原到分支工作树，并核对 snapshot manifest 的配置、代码和文件哈希。
4. 使用 GitHub 分支 `research/alphazero-g-v4` 的同一 commit 代码执行：

   ```sh
   python scripts/run_g_v4.py --advance
   ```

   每次推进一个 10K 里程碑；已完整提交的棋谱按唯一 ID 跳过。若日志多出一局而训练状态尚未原子更新，runner 会从该局补齐优化步。

快照在有效运行时间约 1、3、7、13、23、39、65 小时触发，此后每 26 小时一次。新包只有在上传、回下载、SHA-256 和空目录恢复验证通过后，才允许删除旧 V4 包。
