# V4 实验运行与恢复

在仓库根目录安装研究绘图依赖并依次运行：

```bash
python -m pip install -r requirements-research.txt
python scripts/run_abc_search_v4.py preflight
python scripts/run_abc_search_v4.py run
python scripts/run_abc_search_v4.py analyze
```

`run` 每完成一条记录就追加并 `fsync` 到 `research/v4/results/search_runs.jsonl`，更新原子进度文件；重新执行会按唯一 run ID 跳过已完成记录。`--max-new-runs N` 可做小规模连通性试跑，正式矩阵仍为配置中的 252 个 ID。

创建本地恢复 ZIP：

```bash
python scripts/run_abc_search_v4.py snapshot
```

恢复快照使用 `src.research.abc_search_v4_snapshot.restore_snapshot()`。Drive 备份由执行工作流调用 Drive connector 上传；先下载回本地并校验 SHA-256、运行恢复，再替换上一份正式快照。压缩包只含配置、进度、原始 run JSONL、代码版本和校验清单。
