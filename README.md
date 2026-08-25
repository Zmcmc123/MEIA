[中文](README.md) | [English](README.en.md)

# MEIA

**Molecular and Extended-system Illustration Assistant**

Xiaomei_974 & codex

MEIA 是用于原子构型交互检查和出版级二维示意图制作的可视化应用。软件使用 ASE 读取构型，通过 Streamlit、Plotly 和 Matplotlib 完成交互、渲染与导出。

> MEIA 仍在开发中，不替代专业原子建模软件，也不保证物理结果正确。

## 功能概览

- 中英文界面切换，语言选择保存在浏览器本地。
- 通过 ASE 读取 POSCAR/CONTCAR、CIF、XYZ 和 LAMMPS data 等构型文件。
- 三维旋转、平移、缩放和原子选择，并可对部分原子/化学键进行隐藏/显示操作。
- 共价半径与相等半径两套尺寸方案，支持元素颜色、化学键、氢键、晶胞和周期性设置。
- 大体系按需生成 2D 预览，支持分页原子选择和交互期图层简化。
- 通用风格预设、工作状态快照和批量渲染。
- 导出 SVG、PNG 和 PDF；SVG 保留可编辑分组。

## 下载项目

1. 在 GitHub 仓库页面点击绿色 **Code** 按钮，选择 **Download ZIP**。
2. 解压下载文件，得到通常名为 `MEIA-main` 的项目文件夹。
3. 在终端（Windows 可使用 Anaconda Prompt）进入该目录，确认其中包含 `app.py` 和 `requirements.txt`。

## 环境要求

- Python 3.10（已验证 3.10.19）。
- Python 依赖见 [`requirements.txt`](requirements.txt)。
- 普通运行不需要 Node.js；重建三维前端需要 Node.js 18+ 和 npm。

## 第一次配置环境

### 电脑中还没有 Conda

按 [Miniconda 官方说明](https://docs.conda.io/miniconda.html) 安装对应系统版本。macOS 请区分 Apple Silicon（`arm64`）和 Intel（`x86_64`）。安装后重新打开终端并检查：

```bash
conda --version
```

输出版本号即表示安装成功。

### 创建 MEIA 专用环境

在项目根目录执行：

```bash
conda create -n meia_env python=3.10 -y
conda activate meia_env
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

这会创建 `meia_env` 环境并安装所需依赖。如果已有合适的 Python 3.10 环境，可直接在该环境中执行 `python -m pip install -r requirements.txt`。

## 启动软件

在项目根目录运行：

```bash
conda activate meia_env
python -m streamlit run app.py
```

Streamlit 通常会自动打开浏览器；也可访问 [http://localhost:8501](http://localhost:8501)。按 `Ctrl+C` 停止应用。

从导入示例 `CONTCAR` 到导出 SVG 和工作状态快照的完整流程：

- [中文教程](docs/TUTORIAL.zh-CN.md)
- [English Tutorial](docs/TUTORIAL.en.md)

教程图片使用仓库相对路径并由自动检查校验。如果 GitHub 页面只显示图片占位符，通常是当前网络无法访问 GitHub 的原始图片域名；可下载 ZIP 后直接查看 `docs/images/` 中的图片。

查看批处理参数：

```bash
python -m meia.batch --help
```

## 大体系行为

- 小体系仍自动生成 2D 预览；达到复杂度阈值后，需点击“生成/更新 2D 预览”。已过期预览可供对照，但不能作为当前图像下载。
- 2D 和所有 SVG/PNG/PDF 最终导出始终使用完整数据，不使用降采样结果。
- 源原子数达到 1,000 后，侧栏按每页 200 个原子选择；也可继续使用 3D 点选、框选、序号范围和元素选择。
- 显示实例达到 20,000 后，旋转或缩放期间临时隐藏化学键描边和氢键，停止交互后自动恢复。周期展开的安全上限为 50,000 个原子实例。

开发者可使用 `python -m meia.benchmark_large_system --help` 在本地生成固定随机种子的 slab + 水体系并运行性能基准。仓库不再附带大体积结构或快照。

## 可选：开发三维前端

修改三维交互组件后运行：

```bash
cd meia/components/atom_viewer/frontend
npm ci
npm test
npm run build
```

## 验证

日常修改可先运行快速检查：

```bash
python -m pytest -W error::FutureWarning --strict-markers -q -m "not release"
```

发布前运行完整检查（包含参考产物再生与仓库一致性检查）：

```bash
python -m compileall -q app.py meia tests
python -m pytest -W error::FutureWarning --strict-markers -q
```

详细需求和开发进展见 [`docs/SPEC.md`](docs/SPEC.md) 和 [`docs/PROGRESS.md`](docs/PROGRESS.md)。

## 许可与商业使用

MEIA 是源码可用（source-available）软件，根据 [PolyForm Noncommercial License 1.0.0](LICENSE.md) 提供：

- 个人学习、业余项目和无预期商业应用的研究实验可免费使用。
- 慈善机构、教育机构、公共科研机构、公共安全或卫生机构、环境保护机构和政府机构可按许可条款免费使用。
- 企业、个体经营者或其他商业组织的任何业务相关使用，包括内部科研，均需要单独的书面商业许可。详见 [`license/COMMERCIAL.md`](license/COMMERCIAL.md)。

第三方依赖和已打包前端代码仍遵循各自的许可，详见 [`license/THIRD_PARTY_NOTICES.md`](license/THIRD_PARTY_NOTICES.md)。

问题报告和功能建议请参阅 [`CONTRIBUTING.md`](CONTRIBUTING.md)；安全漏洞请按 [`SECURITY.md`](SECURITY.md) 私下报告。目前暂不接受未经邀请的代码 Pull Request。

Copyright 2026 Xiaomei_974
