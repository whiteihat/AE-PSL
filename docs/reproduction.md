# CIFAR-100 复现入口

批准协议为 `cifar100-cmc-vitb32-split5-ae64-v1`，参数保存在
[`configs/cifar100-reproduction.json`](../configs/cifar100-reproduction.json)。
设置、选择及实际执行结果维护在本人仓库
[`whiteihat/AE-PSL#1`](https://github.com/whiteihat/AE-PSL/issues/1)。
本批覆盖 ImageNet100 GA、CIFAR-100、五客户端、AE-PSL 的一个论文配置。

## 环境与输入

项目 `.venv/` 是 Python 3.12.4 的独立 Conda prefix 环境。普通依赖使用清华
PyPI，torch 2.8.0+cu126 / torchvision 0.23.0+cu126 使用带固定 SHA256 的南京大学
镜像 wheel。已有环境可执行 `bash scripts/install_domestic.sh` 检查或补齐依赖。
该安装入口不改写用户的全局 pip 配置。

ViT-B/32 的 torchvision IMAGENET1K_V1 权重位于
`data/torch/hub/checkpoints/vit_b_32-d86f8d99.pth`，完整 SHA256 为
`d86f8d99b3c385515d1111e2a55e6f52347e8766bb4f29375eccf89fa6bbadb4`。
运行时固定 `TORCH_HOME` 指向本项目 `data/torch`。

| 数据 | 位置与检查 |
|---|---|
| CIFAR-100 原始缓存 | `data/raw/cifar100/cifar-100-python` 软链接到已核对官方 train/test/meta MD5 的 PipeSFL 原始缓存；原文件保留 |
| CIFAR-100 实验输入 | `data/datasets/cifar100/manifest.json`；50,000 train / 10,000 test；PNG 保存原始 32×32 像素；逐图解码与 SHA256 已验证 |
| ImageNet100 镜像 | `asafaa/imagenet100-cmc`，revision `6cc705b814f3a19af8e801d25bb55995ef0ce13e`；十个 ZIP 的 LFS SHA256 已核对 |
| ImageNet100 JPEG | `data/raw/imagenet100/{train,val}`；126,689 train / 5,000 val；100 类、逐类计数和逐图完整解码已检查；不重新缩放或编码 |
| ImageNet100 加载路径 | `data/datasets/imagenet100/train` 指向原 train，`test` 指向原 val；直接使用源码 ImageFolder |
| 类别、映射和指纹 | `data/provenance/imagenet100-cmc/` 中的类别名单、标签、逐图 SHA256 JSONL、归档列表和验证结果 |
| CIFAR-100 来源 | `data/provenance/cifar100-source.json` |

该 ImageNet100 镜像的 CMC 类别名单已固定；AE-PSL 作者未公开足以确认相同名单的
信息。镜像与官方 ImageNet 原始文件未逐字节比较。镜像自身存在五张 train/val 内容
完全相同的 JPEG；保持发布方原始划分，路径见 `validation.json`。
GA 的 val 只报告重建 MSE，不用于本批早停或选取模型。

## 已批准批次

```bash
# 仅解析配置，不加载模型、不训练
.venv/bin/python scripts/run_recorded_experiment.py \
  --mode trial --seed 42 --gpu 0 --dry-run

# 此命令创建全新的 attempts，并启动真实训练：
# 先 seed42/GPU0 全数据 GA1+CSA1+DFT1；
# 验收后 seeds42/43/44 在 GPU0/1/2 并行 GA30+CSA1+DFT10。
.venv/bin/python scripts/run_reproduction_batch.py
```

正在运行的批次无需再次执行上述启动命令。每次新启动都会创建独立目录，不自动重训失败
attempt。仅在已有 trial 成功、而正式批次尚未启动时，可用
`--continue-formal-from /absolute/path/to/successful/trial` 继续已授权的正式批次。

正式各 seed 使用官方主干和新 AE 初始化，`ae_force_retrain=true`，各自拥有独立 AE
权重目录。特征缓存复用遵循源码完整签名；seed42 可使用试跑产生的相同冻结特征，
不读取试跑训练后的 AE 或分类检查点。LoRA rank16/alpha32，split5，latent64，
CLS 不压缩，CIFAR 全局评估绕过 AE。

CSA/DFT 保留源码的按客户端取样顺序与不丢弃尾批行为。每客户端每次最多25张，
完整全局批最多125张；部分客户端耗尽后全局批会变小。
每个 DFT epoch 遍历全部50,000张训练图。DataLoader 不额外重排联邦采样器。
每 seed 的实际客户端数量、尾批、总 batch 数及分区指纹写入 manifest/events。

| seed | GPU | 各客户端样本数，CID0–4 | 每轮全局 batch 数 |
|---|---|---|---|
| 42 | 0 | 11609 / 8734 / 8646 / 12081 / 8930 | 484 |
| 43 | 1 | 10095 / 9153 / 9285 / 10273 / 11194 | 448 |
| 44 | 2 | 8849 / 10066 / 7660 / 10177 / 13248 | 530 |

## 记录与验收

`run_recorded_experiment.py` 调用原始 orchestrator，
`reproduction_recording.py` 在函数返回边界观察真实指标与参数哈希；
原始 `src/ae_psl` 算法文件保持上游版本。记录器来自同一工作区
`research-workspace/src/research_workspace`，无需新增训练依赖。

每个 attempt 包含共享格式的 manifest、metrics、events、status、CSV、
环境版本、GPU、真实命令、源码摘要、控制台日志和原生逐轮 JSON。
GA/CSA/DFT 的 epoch 分别记录，原生 epoch 从0开始，共享曲线中从1开始。
共享 `step` 依次串联三个阶段，准确率单位是比例。
原生训练函数返回的交叉熵另存 `dft/train_loss`、`dft/test_loss`。

验收包含 GA 更新 AE 且主干哈希不变、CSA 更新本地 AE 且主干/LoRA/分类头不变、
服务端 decoder 等于客户端 decoder 均值、DFT 首轮冻结主干/AE 且 LoRA/分类头更新、
完整轮数、完整训练样本、原生成功状态和最终检查点全部张量有限。
DFT 每轮评估10,000张 test；正常退出还执行源码最终评估。
best 是逐轮 test accuracy 的首个最大值，仅用于汇报；final 是最终模型评估，
不加载 best。三 seed 汇总使用 final 的均值和样本标准差。

试跑位于 `tmp/cifar100-trial/`，正式位于 `results/cifar100-formal/`，
监督器队列和启动日志位于 `tmp/cifar100-supervisor/`。
以上输出、全部数据、环境和缓存均被 Git 忽略。
检查点是源码聚合的 client/server 模型权重，不承诺包含完整 optimizer/RNG 恢复状态。

完成批次的曲线可用独立报表环境生成，不给训练环境加绘图库：

```bash
.venv/bin/python -m venv tmp/reporting-venv
tmp/reporting-venv/bin/python -m pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --cache-dir /home/zhoukai/code/.pip-cache matplotlib
tmp/reporting-venv/bin/python scripts/plot_reproduction.py \
  results/cifar100-formal/<batch-id>
```

命令默认在 `curves/` 新建 CSV、PNG 和 PDF；可通过 `--output-directory` 指定新目录
（相对路径以当前工作目录为基准），现有目录不会覆盖。第三方字体缓存及 Matplotlib
配置留在 `tmp/`。本次图例修订的命令为：

```bash
tmp/reporting-venv/bin/python scripts/plot_reproduction.py \
  results/cifar100-formal/20261111T063245Z-81af052c \
  --output-directory results/cifar100-formal/20261111T063245Z-81af052c/curves
```

每个面板包含六项图例，在面板下方按两列三行排列；每行对应一个 seed，左列为实线，
右列为虚线。蓝／橙／绿分别对应 seed 42／43／44。图例使用实际曲线对象生成样线。

| 面板 | 实线与左列图例 | 虚线与右列图例 |
|---|---|---|
| GA：CMC ImageNet100 | `Validation MSE`，验证集重建 MSE | `Train MSE`，训练集重建 MSE |
| DFT：CIFAR-100 | `Test accuracy`，测试集准确率 | `Train accuracy`，训练集准确率 |

GA 的原始记录字段 `ga/test_mse` 对应 ImageNet100 验证集，CSV 用 `stage=ga`
与 `test_mse` 列保存该值，图例明确标为 validation。GA 纵轴为对数 MSE，DFT 纵轴为百分比准确率；
两个横轴均为各阶段 epoch。

## 正式结果与论文位置

在 ViT-B/32、五客户端、split 5、latent 64、CLS 不压缩的条件下，
比较本批 AE-PSL 三 seed 的 final 测试准确率与论文对应配置的三 seed 均值。

| 项目 | 对应位置／实际值 |
|---|---|
| 本批协议 | `cifar100-cmc-vitb32-split5-ae64-v1` |
| 论文版本 | arXiv:2607.17913v1，2026-07-20；项目 `paper/` 中的本地 PDF |
| 直接对应项 | 第 5.3.1 节 Accuracy–Communication Trade-off，Table 1：CIFAR-100 / N=5 / AE-PSL / Mid（R≈10） |
| 压缩档依据 | 附录 A.1，Table 3：Mid 对应 latent=64、CLS 不压缩 |
| 标准差出处 | 附录 A.2，Table 4 的同一单元格 |
| 相关图 | Fig. 2(a)；附录 Fig. 5(a)，均为 CIFAR-100 / N=5 的准确率–通信量图 |
| 对照排除项 | 第 5.3.4 节 Table 2 为 R=15，与本批压缩档不同 |
| 本批汇总口径 | final 均值及样本标准差（ddof=1）；best 单独列出 |

| seed | GA / CSA / DFT 完成 epochs | best 测试准确率 | best epoch | final 测试准确率 | final epoch |
|---|---|---|---|---|---|
| 42 | 30 / 1 / 10 | 82.20% | 10 | 82.20% | 10 |
| 43 | 30 / 1 / 10 | 82.97% | 8 | 82.74% | 10 |
| 44 | 30 / 1 / 10 | 82.53% | 10 | 82.53% | 10 |

| 比较项 | 数值 |
|---|---|
| 本批 final 均值 ± 样本标准差 | 82.49% ± 0.27 个百分点 |
| 论文 Table 4 均值 ± 所报标准差 | 82.7% ± 0.2 个百分点 |
| 本批均值 − 论文均值 | −0.21 个百分点 |
| 绝对均值差 | 约 0.21 个百分点 |
| 相对均值差，以论文均值为分母 | 约 −0.254% |

差值使用论文已公布的一位小数均值计算。作者的 ImageNet100 类别划分、具体 seeds、
表格采用 final 还是 best，以及标准差采用的分母仍未全部确认。本批图的横轴是 epoch；
论文 Fig. 2(a) 和 Fig. 5(a) 的横轴是通信量，本次未计算整条通信曲线的复现误差。

汇总来源为
[`summary.json`](../results/cifar100-formal/20261111T063245Z-81af052c/summary.json)，
修订图、CSV 与逐项对照说明位于
[`curves/`](../results/cifar100-formal/20261111T063245Z-81af052c/curves/)。
旧版图表已删除，修订版统一使用 `curves/`；原始训练记录保留。

数据下载和环境问题按用户最新授权自行处理、保留证据；实验运行若发现需要修补源码的
bug，监督器停止相关批次，保留失败记录后与用户 grill，不自动改算法或重跑。
上述批次执行时未提交或推送本地更改。源码和轻量证据按 2026-11-18 用户授权整理发布，见 [research-workspace #6](https://github.com/whiteihat/research-workspace/issues/6)。

## 源码支持的模型

主干枚举目前只有 `vit_b_32`；对应 torchvision ViT-B/32 的 ImageNet-1K 预训练权重。
本批已确定使用该主干与 `2_layer_MLP_(d_d_d_z)`，不再重新选择。

源码 AE 注册表完整名单：

```text
identity
1_layer_MLP_(d_d_z)
1_layer_MLP_(d_d_z)_linear
2_layer_MLP_(d_d_4_d_z)
2_layer_MLP_(d_d_d_z)
2_layer_MLP_(d_d_d_z)_linear
2_layer_MLP_(d_d_d_z)_ReLU
2_layer_MLP_(d_d_5_4_d_z)
double_linear
3_layer_MLP_(d_d_2_d_4_d_z)
1_layer_CONV
```

论文副本在 `paper/`，工作区根目录的原文件保留。
