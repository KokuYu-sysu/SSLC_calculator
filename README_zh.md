# SSLC 风险计算器

> **[English Version (README.md)](README.md)**

用于批量计算临床 CSV 表格人群的肺癌死亡风险。随附的 `models.pkl` 已替换为更新后的 `model_groups_dict_de_compact.pkl`，包含四个性别/吸烟亚组各自的模型、标准化器和有序特征列表。本文件夹可独立运行，无需网站代码。

## 安装与运行

在你想要安装的目录下打开powershell，并通过以下方式进行下载：

```powershell
git clone https://github.com/KokuYu-sysu/SSLC_calculator.git
cd SSLC_calculator
```

> 注意，为了防止与你的总的Python环境发生冲突，建议单独建立虚拟环境进行隔离。

```powershell
python -m pip install -r requirements.txt
python batch_risk.py input.csv
```

默认计算 5 年风险，生成三个文件：

- `input_with_risk_5y.csv`：保留原始所有列，在最后新增 `risk_5y`；
- `input_errors.csv`：每个无效变量各占一行的错误报告；
- `input_schema_report.json`：输入列与模型/转换规则的一致性报告。

指定其他时间窗，例如 7 年：

```powershell
python batch_risk.py input.csv --time 7
```

结果列将为 `risk_7y`。更新后的模型共同支持 **1.4839228235646376–10.55532786885245 年**（含端点），因此 5、7、10 年均可使用。脚本从拟合模型自身读取时间范围，XGBSE 使用 `model.time_bins`；超过共同范围会在计算前停止。小数时间窗也可使用，例如 `--time 7.5` 输出列为 `risk_7_5y`。CoxNet 使用其拟合生存阶梯函数；XGBSE 使用不晚于指定年份的最近一个时间点，不进行插值。

风险值为 0–1 的概率，并非百分数：`0.01` 表示指定时间内肺癌死亡风险为 1%。

## 输入格式

 `input_template.csv`作为一个模板可以进行参考，填入数据后另存为 CSV。每行可属于不同的性别和吸烟亚组。`sex` 必填且只能为 `female` 或 `male`；`smoking` 必填且只能为`never` 或 `ever`。当 `smoking=ever` 时，`smoking_status` 必填，且为`former` 或 `current`。

更新后的模型比旧版需要更多输入变量。使用分类形式输入时，每行需填写下表的公共字段。女性还需填写 `first_period_age`、`menopause`、`first_live_birth_age` 和 `breastfeeding`，男性可将这四项留空。吸烟者还需填写 `years_smoking` 和 `cigarettes_per_day`；既往吸烟者必须填写 `years_since_quit`，当前吸烟者可将其留空或填 0。

从不吸烟者可将 `smoking_status`、`years_smoking`、`cigarettes_per_day` 和 `years_since_quit` 留空，模型所需的对应吸烟量/年数自动生成 0；若这些模型输入中明确填写非零值，则该行报错。当前吸烟者无论男女，戒烟年数都自动生成 0，明确填写非零值也会报错。疾病史等其他测量变量缺失时，不会自动补 0。

| 字段 | 允许值/范围 | 说明 |
| --- | --- | --- |
| `age` | 40–80 | 年龄 |
| `education` | 整数 1–6 | 1 未接受正规教育；2 小学；3 初中；4 高中/职业学校；5 大专；6 本科及以上 |
| `occupation` | 整数 1–10 | 1 专业技术人员；2 管理人员；3 商业人员；4 农民；5 办事员/工人；6 服务人员；7 家务；8 退休；9 无业；10 其他 |
| `marital` | `married`、`unmarried`、`others` | 婚姻状况 |
| `alcohol` | `never`、`former`、`current` | 饮酒状态 |
| `height`、`weight` | 分别为 `(0,250]` cm、`(0,200]` kg | 用于计算BMI |
| `years_smoking`、`cigarettes_per_day`、`years_since_quit`、`passive_smoking_years` | 不小于 0 | 吸烟和被动吸烟年数；支/日自动换算为包/日 |
| `cancer_history`、`copd`、`chronic_resp`、`diabetes`、`hypertension`、`chd`、`liver`、`tb`、`exercise` | `yes` 或 `no` | 疾病及规律运动 |
| `family_cancer_count` | 整数 0–20 | 患癌一级亲属人数 |
| `pickled`、`protein`、`vegetable` | `never`、`sometimes`、`frequently` | 从不为 0；其余为 1 |
| `oil_smoke` | `none`、`low`、`moderate`、`high` | 无/少许为 0；较多/很多为 1 |
| `first_period_age` | 0–80 的有限数值 | 女性初潮年龄，单位为年；映射为 `first_period_age_new` |
| `menopause` | `premenopausal`、`<40`、`40-45`、`45-50`、`50-55`、`>=55` | 女性绝经年龄 |
| `first_live_birth_age` | `nulliparous`、`<20`、`20-25`、`25-30`、`>=30` | 女性首次活产年龄；映射为 `live_birth_age_*` 独热列 |
| `breastfeeding` | `nulliparous`、`parous_none`、`<6`、`6-12`、`>=12` | 总哺乳时长；前两项均映射到同一模型类别 |

分类输入中，`first_live_birth_age=nulliparous`（未生育）必须对应 `breastfeeding=nulliparous`；已生育者应选择 `parous_none`（未哺乳）、`<6`、`6-12` 或 `>=12`。模型中的 `bfeed_1` 同时表示未生育和已生育但未哺乳，因此仅凭这一独热列无法区分两者。

临床表格分类应遵循研究定义：`smoking=never` 包含一生吸烟少于 100 支的偶尔吸烟者；`alcohol=never` 包含每周饮酒少于一次；规律运动指平均每周至少三次、每次 30 分钟。`chd` 包括冠心病、心衰、中风、周围血管疾病、风湿性/先天性心脏病及心肌病；`family_cancer_count` 中的一级亲属为父母、子女和兄弟姐妹。

## 已独热编码的表格

脚本也接受 `BMI_new`、`education.y`、`first_period_age_new` 等数值型模型输入，以及完整的 `occupation_*`、`marital_*`、`alcohol_*`、`smoking_*`、`menopause_age_*`、`live_birth_age_*` 和 `bfeed_*` 独热列。仍需提供 `sex`、`smoking` 以确定亚组，吸烟者仍需提供 `smoking_status`。脚本核对标准化器与模型实际保存的特征名和顺序，先按完整的 `input_features` 标准化，再按保存顺序选择 `model_features`。

- 一个分类组里的独热列必须完整、取值只能为 0/1，并且必须互斥；
- 同时提供分类列和独热列时，两种表示必须一致；
- 如果保存的 `input_features` 省略了某组的参考类别列，其余列全 0 可表示该参考类别；若保存的是完整分类组，必须恰好有一个 1。仅从 `model_features` 中省略的列仍是标准化器必需的输入；
- 列名或值不符合规则时，该行风险留空并写入错误报告，不会静默补零。

首次活产年龄映射为：`nulliparous` → `live_birth_age_0`，`<20` → `live_birth_age_20`，`20-25` → `live_birth_age_25`，`25-30` → `live_birth_age_30`，`>=30` → `live_birth_age_30more`。独热输入中的未生育行也必须满足 `bfeed_1=1`。

男性不吸烟模型需要的 `gender` 根据 `sex` 自动生成，编码为女性 1、男性 0，无需在分类 CSV 中额外添加；若为该模型直接提供 `gender`，必须与 `sex` 一致。`smoking_pack_day_new` 的单位已是包/日，只有 `cigarettes_per_day` 会除以 20。

可保留 `id`、`patient_id`、结局或其他协变量列；默认它们原样保留在结果中、不传入模型，但会在 `*_schema_report.json` 标记为 `unknown_columns`。如需强制每一列都被识别：

```powershell
python batch_risk.py input.csv --strict-columns
```

## 输出、错误与安全

部分行无效时，继续计算有效行；无效行仍保留在结果中，风险列留空。错误 CSV 包含 `source_row_number`、`error_code`、`message`、`subgroup` 和 `requested_time_years`；行号包含表头，因此第一条数据对应第 2 行。同一受试者有多处问题时会有多条错误记录。

列名报告在逐行预测前建立，列出已识别列、分类来源列、模型输入列、未知列，以及 `missing_by_subgroup`（各亚组缺失输入）和 `missing_routing_columns`（缺失的亚组选择列）。有列名不等于有有效值：既往吸烟者的戒烟年数及其他亚组约束仍逐行检查。

可以自定义输出路径：

```powershell
python batch_risk.py input.csv --output result.csv --errors problems.csv --schema-report schema.json
```

脚本拒绝覆盖输入文件、已有输出文件或已有同名风险列。输入默认按 `utf-8-sig` 读取；医院系统导出的其他编码可明确指定，例如 `--encoding gb18030`。输出使用 UTF-8 BOM，可直接用 Excel 打开中文。

`models.pkl` 是 Python pickle，只能使用随本工具分发、可信来源的模型文件；不要替换成未知来源的 `.pkl`。

若出现 `[existing_output]：refusing to overwrite an existing output file`，请移动已有输出文件，或为三个输出文件分别指定尚未使用的路径。

## 更多帮助

```powershell
python batch_risk.py --help
```

## 协议

MIT
