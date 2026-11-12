# WuziQ G V6.3 交付包

本目录包含完整原始实验归档的分块文件。归档包含 `research/g_v6_3/` 下的完整数据、G2500/G5000 检查点、训练轨迹、全部480局棋谱、逐局计时/搜索统计、配置、审计、中文报告与图表，以及最终版 `scripts/report_g_v6_3.py`。

## 重组与校验

在仓库根目录运行：

```bash
cat research/g_v6_3/github_bundle/data_bundle.part-* > /tmp/WuziQ-G-V6.3-delivery.zip
echo "b13222567d07919e8abc61bc5dbdf21f045c3438bff50e2632dbb672b83e313f  /tmp/WuziQ-G-V6.3-delivery.zip" | sha256sum -c -
unzip -q /tmp/WuziQ-G-V6.3-delivery.zip -d /tmp/WuziQ-G-V6.3-delivery
```

`SHA256SUMS` 列出完整ZIP与每个分块的哈希。仓库中的中文报告、统计表和图表也以普通文件直接提供，便于浏览。

Google Drive 的 V6.3 专用恢复目录保留最新两份经过下载、哈希和空目录恢复验证的正式快照；链接与状态见 `../results/snapshot_status.json`。
