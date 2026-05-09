# AI4Art 生物模型艺术化：第一阶段（平面作品）方案

> 目标：把**生物数据**（FASTA/PDB）和**生物模型**（ESM/Evo 等）转化为具有“抽象美学 + 超现实风格”的可展览平面图像。

## 1. 创作总框架

我们把流程拆成 4 层：

1. **科学语义层（Data Semantics）**：基因序列、蛋白序列、结构信息、模型中间表征。
2. **幻觉生成层（Hallucination Engine）**：高温采样、噪声注入、跨模型混合、目标函数扰动。
3. **视觉映射层（Visual Mapping）**：把 token/embedding/contact map/structure graph 映射为颜色、纹理、几何、笔触。
4. **策展叙事层（Curatorial Narrative）**：每幅作品附带“科学来源 + 生成参数 + 艺术陈述”。

---

## 2. 针对你的数据类型的可执行创作方向

### A. 基因/蛋白 FASTA → “序列织锦（Sequence Tapestry）”

- **输入**：DNA/Protein FASTA。
- **科学处理**：
  - k-mer 频谱（k=3~6）
  - 复杂度（Shannon entropy）
  - motif 密度 / GC 含量 / 保守位点
- **幻觉策略**：
  - temperature 扫描：0.8 → 1.2 → 1.8
  - top-k / top-p 组合
  - 在 embedding 空间添加各向异性噪声（沿主成分方向）
- **视觉映射**：
  - token→颜色；entropy→亮度；motif→纹理印章；突变位点→裂纹/荧光线
- **输出**：超高分辨率抽象平面（如 8k PNG/TIFF）。

### B. PDB 结构 → “折叠梦境（Folding Dreams）”

- **输入**：PDB 坐标、二级结构、残基接触图。
- **科学处理**：
  - residue contact map
  - distance matrix
  - 二级结构分段（helix/sheet/loop）
- **幻觉策略**：
  - 结构图扰动（边重连、局部弹性形变）
  - 多尺度噪声（原子级 + 域级）
  - 将稳定核心与柔性区域分开采样
- **视觉映射**：
  - helix/sheet/loop 映射为不同笔触语法
  - contact map 作为“城市夜景”几何骨架
  - 距离矩阵频域变换后做“生物星云”纹理
- **输出**：一组系列画（同一蛋白不同“梦境温度”版本）。

### C. 模型中间层（ESM/Evo hidden states）→ “生物意识切片（Latent Consciousness Slices）”

- **输入**：模型层输出 embedding / attention。
- **科学处理**：
  - 按层抽取 token embedding
  - attention head 结构图
  - PCA/UMAP/t-SNE 投影
- **幻觉策略**：
  - 层间插值（layer 8 ↔ layer 24）
  - 多模型 latent 融合（ESM + Evo）
  - loss 改写：最大化“稀有模式激活”
- **视觉映射**：
  - attention 矩阵→分形网格
  - latent trajectory→流体画笔路径
- **输出**：可用于展览墙面的“模型内在视觉语言”系列。

---

## 3. “幻觉引擎”参数字典（建议记录到元数据）

每幅作品建议保存 JSON 元数据：

- `model_name`
- `input_id`
- `temperature`
- `top_k` / `top_p`
- `noise_type`（gaussian/perlin/spectral）
- `noise_scale`
- `mix_ratio`（多模型融合比）
- `loss_hack`（若有）
- `seed`
- `render_style`

这会让作品同时具备：可复现性（科学）+ 不可预期性（艺术）。

---

## 4. 第一批作品（MVP）建议：7 天可落地

1. 选 20 条蛋白序列 + 20 个 PDB。
2. 做 3 条生成管线：
   - FASTA→2D 抽象图
   - PDB contact map→几何抽象图
   - ESM embedding→超现实纹理图
3. 每条管线生成 30 张（不同温度/噪声/seed）。
4. 人工策展挑选每条管线 top-5，合计 15 幅首展草案。

---

## 5. 在你服务器上的执行建议（你可在远程机跑）

由于你有大量数据和模型环境，建议建立以下目录规范（都在工作目录下）：

- `data_links/`：软链接到数据库位置（只读）
- `artifacts/intermediate/`：中间特征（embedding/contact map）
- `artifacts/images/`：最终图像
- `artifacts/metadata/`：每张图对应的参数 JSON
- `configs/`：不同艺术实验配置

注意：保持原始数据库目录只读，不修改外部数据。

---

## 6. 下一步我建议我们一起做的两件事

1. **先锁定一个具体数据子集**（比如某类膜蛋白或某条代谢通路相关蛋白），做风格统一的 mini-series。
2. **定义视觉语言手册**：颜色空间、构图规则、噪声美学，保证“像一个艺术家/实验室”的连续风格。

