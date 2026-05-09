# AI4Art
AI4Art x AI4Sci

## FASTA -> Art MVP

首个可运行管线已提供：`scripts/fasta_art_pipeline.py`

### 1) 本地已有 FASTA 文件
```bash
python3 scripts/fasta_art_pipeline.py \
  --fasta /path/to/your.fasta \
  --temperature 1.6 \
  --noise-scale 0.12
```

### 2) 直接从 NCBI 下载后生成
```bash
python3 scripts/fasta_art_pipeline.py \
  --accessions NM_000546 NP_000537 \
  --db nuccore \
  --download-to artifacts/input/example.fasta \
  --temperature 1.8
```

### 3) 使用 ESM2 (650M) embedding 驱动抽象艺术渲染
先执行环境与模型链接脚本（会在 `artifacts/models/` 下创建软链接）：
```bash
bash scripts/setup_esm2_env.sh \
  /tos-mlp-zgci/lichengping/esm2_t33_650M_UR50D \
  /vepfs-mlp2/mlp-public/lichengping/conda_envs/AI4Art
```

然后运行：
```bash
python3 scripts/fasta_art_pipeline.py \
  --fasta /path/to/protein.fasta \
  --use-esm2 \
  --esm2-model-dir artifacts/models/esm2_t33_650M_UR50D \
  --embedding-device cuda \
  --temperature 1.9 \
  --noise-scale 0.08 \
  --max-grid-size 1536
```

输出：
- 图像：`artifacts/images/fasta_art/*.png`
- 元数据：`artifacts/metadata/*.json`

> 该脚本不会删除任何外部文件；默认只在本仓库 `artifacts/` 下写入。

## 生物学标注（新增）

每张图都会在画面底部标注：`accession/db/seq_type/len/temperature/noise`，并额外输出一个总索引：

- `artifacts/metadata/art_manifest.tsv`

你可以用这个清单在策展时快速追溯到 NCBI 编号和生成参数。
