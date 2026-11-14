# G5000 V6.4 恢复说明

本轮冻结模型：G5000（5000 局、87,999 个训练状态、4,999 次更新）。本轮不训练，也不复制或改写 V6.3 的模型/训练状态。

## 冻结模型来源

- V6.3 Drive 恢复目录：<https://drive.google.com/drive/folders/1u3556ink-VFf9a7AbS701flr0V0spSzQ>
- 已验证恢复包序号：12；SHA-256：`c7a615c112324d955e32122f66f6930254510d111ff23606f7805e58f61af592`
- G5000 节点模型 SHA-256：`cc383f0667cfe02b8e8ccbaecf51eccc2c4f088d56c836a4e91e016cfae3ee97`
- 完整训练状态 SHA-256：`0fb9886068a014d000227ae7b275ff9043322a678d7f259e9730d44ac95398d2`

已从该 V6.3 Drive 包下载并在空目录运行 `scripts/package_g_v6_3_recovery.py --verify`，清单 5,535 个文件均通过 SHA-256 校验；模型、Adam、回放池及 Python/NumPy/PyTorch 随机状态恢复均成功。V6.4 恢复包只保存上述冻结身份，不删除或覆盖 V6.3 原包。

## V6.4 对局恢复

1. 取 Drive 目录 <https://drive.google.com/drive/folders/1cvdyNM-M8RgYwnS7Y4moCRBAbXB_pXIB> 中最新有效的 `v6_4_recovery_*.zip`。
2. 对下载文件运行 `sha256sum`，与该快照旁的 `*.sha256.json` 记录比对。
3. 在空目录运行：

   ```bash
   python scripts/restore_g_v6_4_snapshot.py --archive <snapshot.zip> --destination <empty-dir>
   ```

4. 从 V6.3 Drive 恢复包 #12 运行 `scripts/package_g_v6_3_recovery.py --verify`，再使用完整 V6.3 检查点目录。先核对训练状态及模型 SHA 与本文件一致。
5. 将快照中的 `research/g_v6_4/config/`、`research/g_v6_4/results/` 合并到 V6.4 工作树后继续：

   ```bash
   python scripts/run_g_v6_4.py --checkpoint-dir <verified-reference400_empty-checkpoint-dir>
   ```

   执行器按固定唯一 game ID 校验已完成棋谱，跳过有效对局；缺失对局从固定开局重赛。每局原子落盘，单局搜索状态不恢复。

每份 V6.4 快照包括实验配置、已完成完整棋谱、汇总进度、恢复脚本、代码及 SHA-256 清单。不含训练 checkpoint（本轮没有训练）；V6.3 原恢复包保持独立且不被清理。Drive 目录至少保留最近两份通过下载哈希和空目录恢复验证的 V6.4 快照。
