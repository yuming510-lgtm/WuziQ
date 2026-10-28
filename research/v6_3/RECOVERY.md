# V6.3 恢复入口

Drive 目录：<https://drive.google.com/drive/folders/1JyO_OUqnhrIA5cMslFH_UUhUZQj5aEAO>。目录名为 `WuziQ V6.3 Champion Tiebreak 2026-10-28`；目录内仅保留最新有效的 V6.3 快照。V6/V6.2 快照位于其他目录，不得删除。

恢复步骤：

1. 从上述目录下载最新 ZIP，并与 `research/v6_3/results/drive_recovery.json` 中登记的 SHA-256 对照；该 JSON 中的文件 ID 是快照文件，不是目录 ID。
2. 检出快照 `source_identity.json` 中的 `implementation_commit`，保证代码与压缩包中 `source/` 下保存的版本一致。
3. 在空目录执行：

   ```bash
   python scripts/restore_v6_3_snapshot.py /path/to/v6_3_recovery_latest.zip research/v6_3/recovered
   ```

4. 查看恢复目录中的 `progress.json`、`9x9_games.jsonl` 和 `schedule.json`，然后从恢复目录续跑：

   ```bash
   python scripts/run_v6_3_championship.py --output-dir research/v6_3/recovered --max-games 1
   ```

5. 每次续跑完成一局后重新创建快照、上传并完成下载、SHA-256、空目录恢复和棋谱校验；全部验证成功后才删除旧 V6.3 快照。

被中断的当前对局没有写入 JSONL，会从该局固定开局和种子重赛；已经完整落盘的唯一 game ID 不会再次计分。若运行环境不可用，先保留当前 Drive 快照与 GitHub 分支；重启后按上述步骤恢复。项目无模型训练或训练 checkpoint。
