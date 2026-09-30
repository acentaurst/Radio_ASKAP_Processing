# Radio time-series pipeline (ASKAP)

[中文](#中文) | [English](#english)

ASKAP/CASDA 数据下载、动态谱批量生成、射电周期分析，以及可选的 TESS 光变与相位对照。

Scripts for ASKAP/CASDA downloads, batch dynamic-spectrum extraction, radio period searches, and optional TESS light-curve and phase comparisons.

## 中文

### 项目范围

这套代码把 ASKAP 数据下载、Measurement Set（MS）处理和后续时域分析分开。已有 MS 压缩包时，可以直接生成动态谱；已有 `.ds` 文件时，可以跳过下载和成像，只做绘图或周期分析。

目前的下载接口、文件名解析、观测站位置和预处理步骤面向 ASKAP。代码不是支持任意望远镜的一键处理软件，也不是 ASKAP 官方发布的管线。用于其他阵列时，需要重新核对数据格式、馈源与极化约定、校准步骤和时间转换。

各脚本独立运行，主要通过文件顶部的配置区指定输入、输出和分析参数。现有配置包含研究对象与机器相关的默认值，运行前必须修改。

### 目录与入口

复制代码时保留以下层级，尤其是 `Code/science_utils.py`；其他脚本会按相对位置导入它。

```text
Code/
├── science_utils.py
├── 0.Direct_Crossmatch_Proper_motion.py
├── 0.Calculation_Propermotion.py
├── 0.InterProfile_Cleanup.py
├── Data_Downloading/
│   ├── ASKAP_Catalouge_Searching.py
│   ├── ASKAP_Catalogue_Downloading.py
│   ├── ASKAP_FITs_Source_Cutout_Downloading.py
│   ├── ASKAP_MS_Downloading.py
│   └── TESS_Data_Downloading.py
├── Dstools_Pipeline/
│   ├── DS_Batch_Official.py
│   ├── DS_Batch_Official_Matched_SBID.py
│   ├── DS_Batch_Official_VIP.py
│   ├── DS_Rerun.py
│   ├── DS_Plot_IQUV.py
│   ├── DS_Plot.py
│   └── DS_Plot_official.py
└── Period_analysis/
    ├── Radio_LS_PhaseFolding.py
    ├── TESS_Period.py
    ├── TESS_Lightcurve_Phasefolding.py
    └── TESS_Radio_Phase_Validation.py
Processed_Data/
└── Catalogue/                    # 用户准备的 CSV 元数据表
```

`ASKAP_Catalouge_Searching.py` 的拼写沿用现有文件名。

| 脚本 | 用途 |
| --- | --- |
| `ASKAP_Catalouge_Searching.py` | 查询 CASDA 观测元数据，保存 `01.askap_catalogue.csv` |
| `ASKAP_Catalogue_Downloading.py` | 批量下载连续谱分量星表，记录 staging 和下载失败 |
| `0.Direct_Crossmatch_Proper_motion.py` | 将恒星坐标传播到 ASKAP 观测历元，再与本地 XML 星表匹配 |
| `ASKAP_FITs_Source_Cutout_Downloading.py` | 下载 Stokes I/V 完整图像，在本地按自行修正位置截取 FITS 切片 |
| `ASKAP_MS_Downloading.py` | 查询源附近的公开可见度数据，按观测选择最近波束并下载 MS 压缩包 |
| `DS_Batch_Official.py` | 处理所选源目录中的 MS 压缩包，使用传播后的恒星坐标 |
| `DS_Batch_Official_Matched_SBID.py` | 只处理匹配表中的源和 SBID，使用对应 ASKAP 射电分量坐标 |
| `DS_Batch_Official_VIP.py` | 单独处理显式指定的优先 MS 压缩包列表 |
| `DS_Rerun.py` | 在已有工作空间上重新插入、减除模型并提取动态谱；运行前核对其恢复条件 |
| `DS_Plot_IQUV.py` | 本地读取 `.ds`，输出 I/Q/U/V 动态谱和对应数值 CSV |
| `DS_Plot.py` / `DS_Plot_official.py` | 依赖 DStools 的绘图入口；前者可结合 MFS 图像和质量诊断 |
| `Radio_LS_PhaseFolding.py` | 射电 Lomb-Scargle 周期搜索、窗口函数、相位折叠及可选显著性检验 |
| `TESS_Period.py` | 从 TESS LC 测量周期与星历，比较整周计数别名并计算 bootstrap 区间 |
| `TESS_Lightcurve_Phasefolding.py` | TESS 周期图和相位折叠，可读取周期结果或使用手动星历 |
| `TESS_Radio_Phase_Validation.py` | 手动输入周期及误差，比较一份 TESS LC 与一份射电动态谱 |

### 环境准备

下载与本地分析使用 Python 科学计算环境。以下命令安装所需的主要 Python 包，不是锁定版本的环境文件：

```bash
python -m pip install numpy pandas scipy matplotlib astropy astroquery h5py lightkurve statsmodels tqdm keyring keyrings.cryptfile
```

CASDA 下载使用 OPAL 账号。修改各下载脚本中的 `OPAL_USER`，以及查询脚本里 `casda.login(...)` 的用户名；密码通过交互提示或 keyring 提供，不要写进源码。部分脚本显式使用 `keyrings.cryptfile`。认证与 staging 的说明见 [Astroquery CASDA 文档](https://astroquery.readthedocs.io/en/latest/casda/casda.html)。

Astroquery 0.4.11 存在破坏 CASDA 预签名 URL 编码的问题。上游已记录修复，使用前应确认安装版本包含该修复。曾用于本项目 CASDA 下载验证的修订为 `117fb2231`，对应 `0.4.12.dev748+g117fb2231`；这不代表其他依赖也已完成版本锁定。见 [Astroquery 更新记录](https://github.com/astropy/astroquery/blob/main/CHANGES.rst)。需要复现该修订时可在实际运行下载脚本的环境中安装：

```bash
python -m pip install --upgrade "git+https://github.com/astropy/astroquery.git@117fb2231"
python -c "import sys, astroquery; print(sys.executable); print(astroquery.__version__); print(astroquery.__file__)"
```

MS 到动态谱的批处理在 Linux 上运行，另需 DStools、WSClean 和匹配的 CASA/casacore 环境。按 [DStools 上游安装说明](https://github.com/askap-vast/dstools#installation) 配置，避免混用不兼容的系统库与 Conda 库。批处理需要 `dstools-insert-model -p RA DEC -r radius` 的非交互掩模接口；不能假定任意 DStools 发布版本都有这一接口。

在服务器环境中先检查以下命令，再用一个小任务验证实际处理：

```bash
wsclean -version
dstools-askap-preprocess --help
dstools-create-model --help
dstools-insert-model --help
dstools-subtract-model --help
dstools-extract-ds --help
```

`DS_Plot_IQUV.py`、`Radio_LS_PhaseFolding.py` 和 `TESS_Radio_Phase_Validation.py` 直接用 `h5py` 读谱，不要求本地导入 DStools。`DS_Plot.py` 和 `DS_Plot_official.py` 仍需要它。

### 输入表与坐标

两条 DS 路线使用不同坐标，不应混用。

| 路线 | 坐标与必需信息 |
| --- | --- |
| 恒星位置路线 | `hostname`、`ra`、`dec`；自行优先用 `sy_pmra`、`sy_pmdec`，列不存在时才回退到 `pmra`、`pmdec`。结合 `01.askap_catalogue.csv` 的 `obs_id`、`t_min`（MJD UTC）传播到观测历元 |
| 已匹配射电位置路线 | `hostname`、`sbid_clean`、`col_ra_deg_cont`、`col_dec_deg_cont`。坐标为该源、该 SBID 的 ASKAP 射电分量位置，不再做恒星自行传播 |

RA/Dec 为 ICRS 坐标，单位是度；自行是 mas/yr，赤经自行应为包含 `cos(dec)` 的分量。现有恒星路线固定采用 `J2015.5`，不是逐行读取 `epoch0` 的通用传播器。Gaia DR3 的 J2016.0 数据不能直接套用这一设置。缺失自行会触发警告并按缺失分量为零继续，不代表得到了完整的自行修正。

直接交叉匹配脚本还读取 `sy_plx`，单位 mas；有 `default_flag` 时只保留值为 1 的记录。它在每份 XML 中寻找最近射电分量，以默认 3 arcsec 阈值筛选，再按 `hostname + sbid_clean` 去重。这不是基于位置误差的概率匹配，也不会保存搜索半径内所有备选分量。

已匹配路线的最小 CSV 如下，仅用于说明格式，不是真实观测：

```csv
hostname,sbid_clean,col_ra_deg_cont,col_dec_deg_cont
ExampleTarget,12345,150.0,-30.0
```

`sbid_clean` 可用整数、`SB12345` 或 `ASKAP-12345`。若表中有 `col_component_id`，程序也会用它检查歧义。同一源、同一 SBID 的重复行只有坐标与分量标识一致时才合并；不同分量会被跳过，须人工确认。

默认表名沿用原研究项目，并不要求目标一定是系外行星宿主。已有可靠匹配表时可跳过星表下载与自动交叉匹配，但必须满足所选脚本的列要求。旧入口还引用 `_1.csv`、`_2.csv` 等文件名，应分别修改 `INPUT_CSV`，不要假定这些文件已经附带。

### 推荐运行顺序

#### 1. 准备元数据与 MS

没有匹配表时，先配置并运行观测查询、分量星表下载和交叉匹配脚本。两份 CASDA 星表查询目前都写有 `TOP 50000`，不保证覆盖完整档案；需要全量数据时应检查结果是否触及上限并调整查询。

MS 下载器按恒星覆盖范围查找观测，不只下载匹配表列出的 SBID。后续已匹配 DS 入口才按每个源的匹配 SBID 筛选本地包。下载目录须按源组织，文件名保留可解析的 `SB<number>` 与 `beam<number>`。

```text
/data/askap/ms/
└── ExampleTarget/
    └── <CASDA archive containing SB12345 and beam07>.tar
```

FITS 切片不是必需的 DS 输入，但可用于检查交叉匹配。当前切片脚本先下载完整图像再截取，因此应预留网络流量与临时磁盘空间。

#### 2. 先列出已匹配 DS 任务

在 `DS_Batch_Official_Matched_SBID.py` 顶部修改配置，例如：

```python
INPUT_CSV = PROJECT_ROOT / "Processed_Data" / "Catalogue" / "matched_sources.csv"
CASDA_BASE_PATH = "/data/askap/ms"
PIPELINE_RESULTS_BASE = "/data/askap/results/DS"
TARGET_SOURCES = ["ExampleTarget"]
MAX_CONCURRENT_MS = 1
WSCLEAN_THREADS = 8
DRY_RUN = True
```

从仓库根目录运行：

```bash
python Code/Dstools_Pipeline/DS_Batch_Official_Matched_SBID.py
```

`DRY_RUN=True` 只列出源目录、匹配 SBID、beam 和采用的坐标，不解包、不执行 DStools，也不创建结果。此入口只扫描源目录直接下的 `.tar` 文件，不扫描独立 `.ms` 目录，不自动下载缺失数据。`TARGET_SOURCES=[]` 表示处理表中的全部源。

检查 `[QUEUED]` 记录，以及 `[MISSING_MS]`、`[AMBIGUOUS_MATCH]`、`[AMBIGUOUS_MS]` 等提示。相同 SBID/beam 有多个包时程序跳过；不同 beam 可以形成独立任务，需确认这些包确实属于预期数据。

#### 3. 生成动态谱

确认任务后将 `DRY_RUN=False`，先跑一个源的一份压缩包。处理顺序为：

```text
MS archive → unpack → ASKAP preprocess → WSClean model
           → insert model with target mask → subtract model → extract .ds
```

当前提取命令使用 `-u 500 -B`：排除短于 500 m 的基线，并保留基线维度。15 arcsec 的掩模半径和成像参数是现有设置，应根据数据检查，不是所有源的最佳值。并发任务数与每个 WSClean 进程的线程数会共同影响资源占用。

输出按源存放在同一个结果根目录：

```text
<PIPELINE_RESULTS_BASE>/
└── ExampleTarget/
    ├── DS_Results/
    │   ├── ExampleTarget_SB12345_beam7_askap.ds
    │   └── ExampleTarget_SB12345_beam7_askap.ds.coordinates.json
    └── ExampleTarget_SB12345_beam7_askap_workspace/
```

恒星位置路线也使用 `DS_Results/`，文件名没有 `_askap` 后缀。已匹配路线将坐标与若干提取设置写入 JSON；恢复时要求记录一致，否则报 `[COORDINATE_MISMATCH]`。该检查不是完整的配置哈希，也不验证 `.ds` 内容是否可读。保留旧结果，改用新的结果根目录，不要删除 JSON 来强行恢复。

普通运行的日志 `pipeline_execution_matched_sbid.log` 写在当前工作目录。恢复依赖工作空间与 `.wsclean_done`、`.subtraction_done`，不要在处理中清理它们。

#### 4. 本地绘图与射电周期分析

将 `.ds` 复制到本地后，修改 `DS_Plot_IQUV.py` 的 `DS_INPUT_DIR` 或 `SINGLE_DS_FILE`、`BATCH_PROCESS` 和 `OUTPUT_BASE`：

```bash
python Code/Dstools_Pipeline/DS_Plot_IQUV.py
```

批量模式递归查找 `.ds`，输出四个 Stokes 面板的 PNG 和每个时间/频率箱的 CSV。默认频率平均因子为 5；短于 1 小时的观测不做时间平均，其余用因子 6。显示色标裁剪不改变 CSV 数值。已有输出的跳过行为由 `CHECK_LOCAL_EXISTS` 控制。

周期分析另运行：

```bash
python Code/Period_analysis/Radio_LS_PhaseFolding.py
```

先设置 `DS_FILES_DIR`、`TARGET_SBIDS`、输出目录、周期范围和星历。`COMBINE_SBIDS` 控制合并观测；`ENABLE_FAP` 控制解析虚警概率与分块随机检验，默认关闭。脚本还检查窗口函数，并在多历元合并时做逐一剔除历元检验。折叠图用于观察形态，不能单凭图形证明周期显著。

#### 5. 可选的 TESS 分析

下载一维光变产品的例子：

```bash
python Code/Data_Downloading/TESS_Data_Downloading.py --target "TIC 123456789" --type lc --download-dir /data/tess
```

该 TIC 仅为命令格式示例，请替换为真实目标。下载器按 TIC 筛选并检查 FITS，可选下载 TPF 或其他光变产品。`--clean` 会删除目标目录中识别出的非目标 TIC 文件，默认不启用。

`TESS_Period.py` 使用 delivered LC，不以 TPF 作为主周期输入。修改 `TESS_INPUT_PATHS`、`OUTPUT_DIR`、周期范围、分段和别名设置后运行：

```bash
python Code/Period_analysis/TESS_Period.py
```

当前默认搜索范围 `0.165` 至 `0.168` 天是特定目标的配置。输出包括 `TESS_Period_Result.md`、`TESS_Period_Timings.txt`、`TESS_Period_Bootstrap.txt` 和 `TESS_Period.png`，直接写入 `OUTPUT_DIR`。分析不同目标或输入组合时，分别指定输出目录，避免覆盖。

`TESS_Radio_Phase_Validation.py` 采用手动周期，不自动读取上述星历文件。需要设置：

| 配置项 | 单位或含义 |
| --- | --- |
| `TESS_TEMPLATE_FILE`、`RADIO_FILE` | 一份 TESS LC 和一份 `.ds` 的路径 |
| `MANUAL_PERIOD_DAYS` | 周期，天 |
| `MANUAL_PERIOD_ERR_MINUS_S`、`MANUAL_PERIOD_ERR_PLUS_S` | 周期下侧、上侧 1σ 误差，秒 |
| `MANUAL_T0_BJD_TDB` | 完整 BJD_TDB 星历零点，不是 BTJD 或 MJD |
| `MANUAL_T0_ERR_S` | 零点 1σ 误差，秒；零表示条件于固定零点 |
| `OUTPUT_BASE` | 输出根目录，下面按射电源名建子目录 |

```bash
python Code/Period_analysis/TESS_Radio_Phase_Validation.py
```

输出文件为 `<radio_stem>_P<period>_Combined.png` 和 `_PhaseInfo.csv`。误差包络采用独立高斯近似，不含周期与零点协方差、整周别名或光学模板极小值的不确定性。

### 数据检查与结果解释

- 本地读谱的默认时间是自 MJD 0 起的 UTC 秒，频率为 Hz，流量为 Jy。先核对 `.ds` 的 `time`、`frequency`、`flux`、`uvdist` 和馈源信息；不要仅凭扩展名判断兼容。IQUV 绘图入口按线性馈源转换。
- 光学与射电绝对相位比较需要同一 BJD_TDB 星历和可靠目标坐标。相位脚本可从 `.ds` 的 `phasecentre` 读取位置，必要时用 `TARGET_RA_DEG`、`TARGET_DEC_DEG` 覆盖；它们仍固定使用 ASKAP 站址。
- 周期 bootstrap 区间条件于所选别名、模型和数据处理。较小的周期误差不等于排除了所有整周计数别名。非同时观测的形态相似也不等于长期相位锁定。
- `std/sqrt(n)` 是均值标准误；通道或时间相关性会使它偏小。检查 RFI、校准和残余背景，不能把图上的误差条当作周期显著性检验。
- 下载重试与文件存在检查不能保证科学数据正确。MS 下载器检查压缩包大小与 checksum 文件存在，不计算并比对 checksum；星表下载器按已存在文件跳过，不解析校验其内容。

HTTP 403 不一定是账号权限问题。应区分登录、staging 和文件传输失败；成功 staging 后的 `InvalidAccessKeyId` 还可能与 URL 编码问题有关。超时、访问拒绝与签名过期需要分别处理，公开日志中不要包含预签名 URL 的查询参数。

### 复现、清理与公开发布

仓库目前没有锁定版本的安装环境，也不附带完整观测数据。首次使用时保存 Python 与依赖版本、所选脚本配置、输入表版本和一个实际观测的小规模运行结果。可以先用 `python -m compileall -q Code` 检查语法，再完成匹配任务清单检查、单包 MS 处理及一个 `.ds` 的本地读谱验证；语法通过不代表服务器处理链已通过。部分批处理会记录单个任务失败后继续，最终应检查日志和实际产物，不能只看进程退出码。

保留原始 MS 压缩包。批处理会重建不完整工作空间；`DS_Rerun.py` 会在复用已有模型时替换旧 DS，重跑前应备份成果。`0.InterProfile_Cleanup.py` 会删除工作空间内的 MS 和部分 FITS，只按最终 `.ds` 是否存在判断，不检查谱内容，且没有 dry-run。它不属于首次运行步骤，应在验证并备份结果后单独决定是否使用。

公开代码前移除个人账号名、机器路径、私有数据、日志和预签名链接。只复制所需源码和文档；不要直接复制整个研究目录或 `.git` 历史。现有 `.gitignore` 不会自动排除观测数据和结果，可在新仓库中按实际目录加入忽略规则，例如：

```gitignore
Data/
Result/
*.ds
*.ms/
*.tar
*.fits
*.log
.idea/
.env
```

CSV 元数据也应逐一检查是否适合公开。提供示例时使用明确标注的合成记录；发布实际观测结果时保留数据来源和 SBID。当前代码未附项目许可证，维护者应在公开仓库中选择许可证；依赖包与数据的使用条件仍分别适用。

## English

### Scope

This collection separates ASKAP downloads, Measurement Set (MS) reduction, and time-series analysis. Start with the reduction scripts if you already have MS archives, or use the plotting and period scripts directly if you have `.ds` files.

The download services, filename conventions, preprocessing, and observatory coordinates are ASKAP-specific. This is not an official ASKAP pipeline or a turnkey multi-telescope package. Other arrays require checks of data formats, feed and polarization conventions, calibration, and time conversion.

Scripts run independently. Most settings are edited in the configuration block near the top of each file. The supplied defaults refer to particular targets and machines; replace them before running. Keep the directory layout shown above, including `Code/science_utils.py`.

### Script selection

| Task | Entry point |
| --- | --- |
| Query CASDA observation metadata | `Data_Downloading/ASKAP_Catalouge_Searching.py` |
| Download continuum component catalogues | `Data_Downloading/ASKAP_Catalogue_Downloading.py` |
| Cross-match epoch-propagated stellar positions | `0.Direct_Crossmatch_Proper_motion.py` |
| Download images and make local Stokes I/V cutouts | `Data_Downloading/ASKAP_FITs_Source_Cutout_Downloading.py` |
| Download MS archives, selecting the nearest beam per observation | `Data_Downloading/ASKAP_MS_Downloading.py` |
| Reduce local archives using propagated stellar positions | `Dstools_Pipeline/DS_Batch_Official.py` |
| Reduce matched SBIDs using ASKAP radio component positions | `Dstools_Pipeline/DS_Batch_Official_Matched_SBID.py` |
| Process an explicit priority archive list | `Dstools_Pipeline/DS_Batch_Official_VIP.py` |
| Rerun model insertion, subtraction, and extraction in existing workspaces | `Dstools_Pipeline/DS_Rerun.py` |
| Plot I/Q/U/V locally and export numerical CSV data | `Dstools_Pipeline/DS_Plot_IQUV.py` |
| Plot with DStools, optionally including MFS and QC products | `Dstools_Pipeline/DS_Plot.py`, `Dstools_Pipeline/DS_Plot_official.py` |
| Search radio periods and inspect phase folds | `Period_analysis/Radio_LS_PhaseFolding.py` |
| Fit a TESS period and ephemeris | `Period_analysis/TESS_Period.py` |
| Plot TESS periodograms and phase folds | `Period_analysis/TESS_Lightcurve_Phasefolding.py` |
| Compare TESS and radio data using a manual ephemeris | `Period_analysis/TESS_Radio_Phase_Validation.py` |

Paths in this table are relative to `Code/`. The spelling of `ASKAP_Catalouge_Searching.py` follows the existing filename.

### Environments

Install the main Python dependencies in the environment that will run the scripts:

```bash
python -m pip install numpy pandas scipy matplotlib astropy astroquery h5py lightkurve statsmodels tqdm keyring keyrings.cryptfile
```

This is a dependency list, not a version-locked environment. Set your OPAL username in each downloader and in the query script's `casda.login(...)` call. Supply passwords interactively or through keyring, not in source files. Some downloaders select the encrypted `keyrings.cryptfile` backend. See the [Astroquery CASDA documentation](https://astroquery.readthedocs.io/en/latest/casda/casda.html) for authentication and staging.

Astroquery 0.4.11 can corrupt the percent-encoding of CASDA pre-signed download URLs. Use a revision containing the upstream fix. Revision `117fb2231`, reported as `0.4.12.dev748+g117fb2231`, was used for a CASDA download check in this project. The [Astroquery changelog](https://github.com/astropy/astroquery/blob/main/CHANGES.rst) documents the fix. To reproduce that revision:

```bash
python -m pip install --upgrade "git+https://github.com/astropy/astroquery.git@117fb2231"
python -c "import sys, astroquery; print(sys.executable); print(astroquery.__version__); print(astroquery.__file__)"
```

MS reduction runs on Linux with DStools, WSClean, and compatible CASA/casacore libraries. Follow the [DStools installation instructions](https://github.com/askap-vast/dstools#installation). Check that the installed version accepts the non-interactive `dstools-insert-model -p RA DEC -r radius` interface used here; not every release necessarily provides it. Avoid mixing incompatible system and Conda libraries.

Check the command interfaces before processing an observation:

```bash
wsclean -version
dstools-askap-preprocess --help
dstools-create-model --help
dstools-insert-model --help
dstools-subtract-model --help
dstools-extract-ds --help
```

The local `DS_Plot_IQUV.py`, `Radio_LS_PhaseFolding.py`, and `TESS_Radio_Phase_Validation.py` readers use `h5py` without importing DStools. The older `DS_Plot.py` and `DS_Plot_official.py` interfaces still require it.

### Tables and coordinates

Choose the position convention that matches the intended measurement:

- Stellar-position reduction requires `hostname`, `ra`, and `dec`, with proper motions in `sy_pmra` and `sy_pmdec`. Legacy `pmra` and `pmdec` are used only if the corresponding `sy_` columns are absent. Observation epochs come from `obs_id` and `t_min` (MJD UTC) in `01.askap_catalogue.csv`.
- Matched-SBID reduction requires `hostname`, `sbid_clean`, `col_ra_deg_cont`, and `col_dec_deg_cont`. It uses the ASKAP component position for that source and SBID, without stellar proper-motion propagation.

Coordinates are ICRS positions in degrees; proper motions are in mas/yr, with the RA component including `cos(dec)`. The stellar route assumes J2015.5. It does not read a reference epoch from each row, so do not pass J2016.0 Gaia DR3 positions unchanged. Missing proper-motion components produce warnings and a zero-component fallback, not a complete correction.

The automatic cross-match also reads `sy_plx` in mas and selects `default_flag=1` when that column exists. Within each XML catalogue it selects the nearest component, applies the configured angular threshold (3 arcsec by default), and deduplicates by host and SBID. It is not an uncertainty-based probabilistic match and does not retain every nearby alternative.

A minimal matched table is:

```csv
hostname,sbid_clean,col_ra_deg_cont,col_dec_deg_cont
ExampleTarget,12345,150.0,-30.0
```

This is a synthetic schema example, not an observation. SBIDs may be integers or strings such as `SB12345` and `ASKAP-12345`. Optional `col_component_id` participates in ambiguity checks. Repeated rows are merged only when coordinates and component identifiers agree; conflicting matches are skipped for manual review.

The default filenames come from an exoplanet-host study, but the matched reduction does not require a planetary host. An externally prepared table can bypass catalogue download and automatic matching if it satisfies the selected script's schema. Update `INPUT_CSV` in every entry point you use; some older scripts reference `_1.csv` or `_2.csv` variants that are not supplied automatically.

### Workflow

#### 1. Prepare metadata and archives

If needed, run the CASDA metadata query, component-catalogue downloader, and proper-motion cross-match after editing their settings. Both catalogue queries currently use `TOP 50000`; they do not guarantee complete archive coverage. Check for truncation before treating a sample as complete.

The MS downloader searches observations covering the stellar position, rather than downloading only the matched SBIDs. The matched reduction filters the local archives later. Organize archives directly under source folders, preserving filenames containing `SB<number>` and `beam<number>`. Standalone `.ms` directories are not inputs to the matched batch scanner.

Cutouts can help assess counterparts. The current FITS script downloads a full image before making a local cutout, so allow for temporary storage and full-image transfer costs.

#### 2. Inspect a matched reduction queue

Edit `DS_Batch_Official_Matched_SBID.py`, for example:

```python
INPUT_CSV = PROJECT_ROOT / "Processed_Data" / "Catalogue" / "matched_sources.csv"
CASDA_BASE_PATH = "/data/askap/ms"
PIPELINE_RESULTS_BASE = "/data/askap/results/DS"
TARGET_SOURCES = ["ExampleTarget"]
MAX_CONCURRENT_MS = 1
WSCLEAN_THREADS = 8
DRY_RUN = True
```

Run from the repository root:

```bash
python Code/Dstools_Pipeline/DS_Batch_Official_Matched_SBID.py
```

Dry-run mode lists selected jobs and coordinates without unpacking archives, invoking DStools, or creating results. Review `[QUEUED]`, `[MISSING_MS]`, `[AMBIGUOUS_MATCH]`, and `[AMBIGUOUS_MS]` messages. An empty `TARGET_SOURCES` list selects all matched sources. Duplicate archives for one SBID/beam are skipped; different beams can produce separate jobs. Missing archives are not downloaded by this script.

#### 3. Extract dynamic spectra

Set `DRY_RUN=False` after reviewing the queue. Start with one archive before increasing concurrency. The sequence is unpacking, ASKAP preprocessing, WSClean field modelling, model insertion with a target mask, model subtraction, and dynamic-spectrum extraction.

Extraction currently uses `-u 500 -B`, excluding baselines shorter than 500 m and retaining the baseline axis. The 15 arcsec mask and imaging settings are defaults to inspect, not universal optimum values. Worker count and WSClean threads per worker both contribute to resource use.

Matched products are written to:

```text
<PIPELINE_RESULTS_BASE>/<source>/DS_Results/
    <source>_SB<sbid>_beam<beam>_askap.ds
    <source>_SB<sbid>_beam<beam>_askap.ds.coordinates.json
```

The stellar route uses the same source-based directory layout without the `_askap` suffix. The matched route records coordinates and selected extraction settings in JSON and refuses incompatible recovery with `[COORDINATE_MISMATCH]`. This is not a complete configuration hash or an HDF5 integrity check. Keep existing results and use a new output root rather than removing metadata to bypass the check.

Normal runs write `pipeline_execution_matched_sbid.log` in the working directory. Keep workspaces and their `.wsclean_done` and `.subtraction_done` markers while processing is incomplete.

#### 4. Plot and search radio periods

Set the input, batch mode, and output directory in `DS_Plot_IQUV.py`, then run:

```bash
python Code/Dstools_Pipeline/DS_Plot_IQUV.py
```

Batch mode searches recursively for `.ds` files. Outputs are an I/Q/U/V PNG and a CSV of time-frequency bins. Default averaging is a frequency factor of 5 and a time factor of 1 below one hour or 6 otherwise. Color limits do not clip exported values. `CHECK_LOCAL_EXISTS` controls skipping existing outputs.

For radio periods, configure `DS_FILES_DIR`, `TARGET_SBIDS`, period bounds, ephemeris, and output root:

```bash
python Code/Period_analysis/Radio_LS_PhaseFolding.py
```

`COMBINE_SBIDS` controls combined analysis. `ENABLE_FAP`, off by default, gates analytic false-alarm probabilities and block-permutation checks. The script also inspects the sampling window and performs leave-one-epoch-out checks for combined multi-epoch data. A folded profile alone is not a significance test.

#### 5. Add TESS analysis if needed

For a real target, replace the example TIC identifier and download LC products:

```bash
python Code/Data_Downloading/TESS_Data_Downloading.py --target "TIC 123456789" --type lc --download-dir /data/tess
```

The downloader filters by TIC and checks FITS files; TPF and other light-curve products are optional. `--clean` deletes recognized foreign-TIC files from the target directory and is disabled by default.

Configure `TESS_INPUT_PATHS`, `OUTPUT_DIR`, search bounds, timing segments, and aliases in `TESS_Period.py`:

```bash
python Code/Period_analysis/TESS_Period.py
```

Its primary inputs are delivered one-dimensional LC products, not TPFs. The current 0.165 to 0.168 day search range is target-specific. It finds candidate periods, fits segment timings with a weighted ephemeris, compares integer-cycle aliases, and calculates paired bootstrap intervals. Outputs are `TESS_Period_Result.md`, `TESS_Period_Timings.txt`, `TESS_Period_Bootstrap.txt`, and `TESS_Period.png`, written directly to `OUTPUT_DIR`. Use separate directories for different targets or input selections to avoid overwriting results.

For `TESS_Radio_Phase_Validation.py`, supply one LC, one `.ds`, and a manual ephemeris. This entry point does not read the fitted ephemeris files automatically.

| Setting | Meaning |
| --- | --- |
| `TESS_TEMPLATE_FILE`, `RADIO_FILE` | Input LC and dynamic spectrum |
| `MANUAL_PERIOD_DAYS` | Period in days |
| `MANUAL_PERIOD_ERR_MINUS_S`, `MANUAL_PERIOD_ERR_PLUS_S` | Lower and upper 1σ period errors in seconds |
| `MANUAL_T0_BJD_TDB` | Full BJD_TDB reference epoch, not BTJD or MJD |
| `MANUAL_T0_ERR_S` | 1σ epoch error in seconds; zero conditions on a fixed epoch |
| `OUTPUT_BASE` | Output root, with a subdirectory for the radio source |

```bash
python Code/Period_analysis/TESS_Radio_Phase_Validation.py
```

Outputs are `<radio_stem>_P<period>_Combined.png` and `_PhaseInfo.csv`. Its uncertainty envelope uses independent Gaussian errors and excludes period-epoch covariance, cycle-count aliases, and template-minimum uncertainty.

### Checks and interpretation

- Local readers default to UTC seconds since MJD 0, frequency in Hz, and flux in Jy. Inspect `time`, `frequency`, `flux`, `uvdist`, and feed metadata rather than assuming compatibility from the `.ds` suffix. The IQUV plotter uses linear-feed conversion.
- Absolute optical-radio phase comparisons need a shared BJD_TDB ephemeris and reliable target coordinates. Phase scripts can use the `.ds` `phasecentre`, or explicit `TARGET_RA_DEG` and `TARGET_DEC_DEG` overrides where provided. Observatory coordinates remain ASKAP-specific.
- Bootstrap period intervals are conditional on the chosen alias, model, and processing. A narrow interval does not eliminate alternative cycle counts. Similar profiles from non-simultaneous observations do not establish long-term phase locking.
- `std/sqrt(n)` is a standard error of the mean. Correlated channels or integrations can make it optimistic. Inspect RFI, calibration, and residual background separately from period significance.
- Retry and existence checks are not scientific validation. The MS downloader checks archive size and the presence of a nonempty checksum file, but does not compare a computed checksum. The catalogue downloader skips existing files without parsing their contents.

A 403 response is not sufficient evidence of an account-permission problem. Separate login, staging, and transfer failures. An `InvalidAccessKeyId` after successful staging can also arise from URL corruption; timeouts and expired signatures require different checks. Remove signed URL query strings from shared logs.

### Reproducibility and sharing

No version-locked environment or complete observational dataset is bundled. Record the interpreter, package versions, script settings, catalogue versions, and a small real-observation run. `python -m compileall -q Code` checks syntax only. Follow it with queue inspection, one MS reduction, and one local `.ds` read before relying on a batch result. Some batches log a failed task and continue; inspect logs and products rather than relying on the process exit code alone.

Keep original MS archives. Batch processing can rebuild incomplete workspaces. `DS_Rerun.py` replaces existing DS products while reusing its retained model; back up those products before rerunning. `0.InterProfile_Cleanup.py` deletes workspace MS directories and selected FITS files, checks only whether a final `.ds` exists, and has no dry-run mode. Use it only after validating and backing up products, not during initial setup.

Before publishing, remove personal account identifiers, machine paths, private data, logs, and signed links. Copy the intended source and documentation, not the entire research directory or its `.git` history. The current ignore rules do not exclude observational data or results; add rules appropriate to the new repository. CSV metadata also needs review. Use labelled synthetic examples, and retain provenance and SBIDs when sharing real results.

No project license is supplied in this code collection. The maintainer should choose one for the public repository; dependency licenses and archive data terms apply separately.
