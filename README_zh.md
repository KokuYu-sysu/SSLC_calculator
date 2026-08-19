# SSLC 风险计算器

> **[English Version (README.md)](README.md)**

## 安装与运行

在你想要安装的目录下打开powershell，并通过以下方式进行下载：

```powershell
git clone https://github.com/KokuYu-sysu/SSLC_calculator.git
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

结果列将为 `risk_7y`。当前随附模型共同支持约 2-10 年；超过范围会在计算前停止，避免将模型外推成长期风险。小数时间窗也可以使用，例如`--time 7.5` 输出列为 `risk_7_5y`。

## 输入格式

 `input_template.csv`作为一个模板可以进行参考，填入数据后另存为 CSV。每行可属于不同的性别和吸烟亚组。`sex` 必填且只能为 `female` 或 `male`；`smoking` 必填且只能为`never` 或 `ever`。当 `smoking=ever` 时，`smoking_status` 必填，且为`former` 或 `current`。

只需填写当前行所在亚组实际使用的变量；不使用的变量可以留空。

| 字段 | 允许值/范围 | 说明 |
| --- | --- | --- |
| `age` | 40–80 | 年龄 |
| `education` | 整数 1–6 | 教育程度 |
| `occupation` | 整数 1–10 | 1 专业技术人员；2 管理人员；3 商业人员；4 农民；5 办事员/工人；6 服务人员；7 家务；8 退休；9 无业；10 其他 |
| `marital` | `married`、`unmarried`、`others` | 婚姻状况 |
| `alcohol` | `never`、`former`、`current` | 饮酒状态 |
| `height`、`weight` | 分别为 `(0,250]` cm、`(0,200]` kg | 用于计算BMI |
| `years_smoking`、`cigarettes_per_day`、`years_since_quit`、`passive_smoking_years` | 不小于 0 | 吸烟和被动吸烟年数；支/日自动换算为包/日 |
| `cancer_history`、`copd`、`chronic_resp`、`diabetes`、`hypertension`、`chd`、`liver`、`tb`、`exercise` | `yes` 或 `no` | 疾病及规律运动 |
| `family_cancer_count` | 整数 0–20 | 患癌一级亲属人数 |
| `pickled`、`protein`、`vegetable` | `never`、`sometimes`、`frequently` | 从不为 0；其余为 1 |
| `oil_smoke` | `none`、`low`、`moderate`、`high` | 无/少许为 0；较多/很多为 1 |
| `menopause` | `premenopausal`、`<40`、`40-45`、`45-50`、`50-55`、`>=55` | 女性绝经年龄 |
| `breastfeeding` | `nulliparous`、`parous_none`、`<6`、`6-12`、`>=12` | 总哺乳时长；前两项均映射到同一模型类别 |

## 已独热编码的表格

脚本也接受模型变量名已经存在的表格，例如 `occupation_1`、`occupation_63`、`marital_have`、`alcohol_no`、`menopause_age_50` 或 `bfeed_12`。这些列会在传入模型前与 PKL 内 `input_features` 严格核对。

- 一个分类组里的独热列必须完整、取值只能为 0/1，并且必须互斥；
- 同时提供分类列和独热列时，两种表示必须一致；
- 包含模型未使用的参考类别时，全 0 表示该参考类别；
- 列名或值不符合规则时，该行风险留空并写入错误报告，不会静默补零。

可保留 `patient_id`、结局或其他协变量列；默认它们不传入模型，但会在`*_schema_report.json` 标记为 `unknown_columns`。如需强制每一列都被识别：

```powershell
python batch_risk.py input.csv --strict-columns
```

## 输出、错误与安全

错误 CSV 固定包含 `source_row_number`、`error_code`、`message`、`subgroup` 和`requested_time_years`。同一受试者有多处问题时会有多行，便于一次性修正全部变量。

可以自定义输出路径：

```powershell
python batch_risk.py input.csv --output result.csv --errors problems.csv --schema-report schema.json
```

脚本拒绝覆盖输入文件、已有输出文件或已有同名风险列。输入默认按 `utf-8-sig` 读取；医院系统导出的其他编码可明确指定，例如 `--encoding gb18030`。输出使用 UTF-8 BOM，可直接用 Excel 打开中文。

`models.pkl` 是 Python pickle，只能使用随本工具分发、可信来源的模型文件；不要替换成未知来源的 `.pkl`。

如果出现错误： `[existing_output]：refusing to overwrite an existing output file` 需要将已生成的文件删除后重新运行

## 更多帮助

```powershell
python batch_risk.py --help
```

## 协议

MIT
