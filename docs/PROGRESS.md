# MEIA 项目进展报告

> 日期：2026-08-25
> 阶段：v0.11.0 功能基线、公开仓库加固与大体系优化已完成本地验证

## 1. v0.11.0 当前能力

| 能力           | 当前实现                                                                                             | 主要位置                                                                          |
| ------------ | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| 双语主界面        | 支持中文与英文一键切换；中文标题保持“⚛ 原子构型可视化”，英文标题为“MEIA - Molecular and Extended-system Illustration Assistant” | `app.py`, `meia/i18n.py`, `meia/locales/`                                     |
| 语言优先级        | 手动选择高于已保存 Cookie，已保存 Cookie 高于浏览器请求语言；手动选择通过 `meia.locale` 持久化，首次加载不依赖自定义静态组件 | `app.py`, `meia/locale_state.py`                                              |
| 3D 交互双语      | 视角、轴向、选择、错误、无障碍标签和 Plotly 图例全部由词典提供；英文成键术语统一为 `Detected Bonds` / `Show Detected Bonds`           | `meia/viewer.py`, `meia/components/atom_viewer/`                              |
| 切换状态保留       | 语言切换前的结构、侧栏未提交草稿、3D 相机方向、缩放、角度步长、选择模式和临时选区保留；结构或视图修订变化时自动隔离旧状态                                   | `app.py`, `meia/components/atom_viewer/frontend/src/viewer-session-state.mjs` |
| 可追溯双语错误      | 预设、构型读取、原子尺寸/颜色、键范围/描边、氢键阈值、周期范围/上限、选择和相机均携带稳定翻译键与参数；未知第三方错误保留原始详情和用户路径                          | `meia/i18n.py`, `meia/presets.py`, `meia/io.py`, visual-domain modules        |
| 双尺寸档案        | 共价半径与相等半径各自保存全局倍率、元素覆盖和化学键粗细；默认分别为 `0.6 / 0.45` 与 `1.0 / 0.35 Å / 0.45`                          | `meia/size_profiles.py`, `meia/sidebar.py`                                    |
| 原子化模式应用      | 下拉切换只修改草稿；点击“应用原子设置”后，目标档案的原子半径与化学键粗细同时生效                                                        | `app.py`, `meia/sidebar.py`, `meia/visual_state.py`                           |
| 跨视图半径一致      | 单一 `RenderContext` 解析最终半径，3D、2D、SVG/PNG/PDF、化学键与氢键表面裁剪共用                                         | `meia/config.py`, `meia/projection.py`, `meia/viewer.py`, `meia/renderer.py`  |
| 一键还原         | 按钮位于侧栏最底部；恢复原子/键/晶胞与周期性/选择，清除已作废草稿，保留相机、导出、结构和快照会话                                               | `app.py`, `meia/sidebar.py`, `meia/visual_state.py`                           |
| 严格 schema v7 | 顶层 `size_profiles` 保存当前模式和两套完整档案；当前元数据使用 `meia_version`，旧版与未来版明确拒绝                               | `meia/presets.py`, `meia/workspace.py`, `meia/batch.py`                       |
| MEIA 持久化产物   | 通用风格和工作状态分别使用 `.style.meia.json` 与 `.workspace.meia.json`，默认导出名为 `meia-visual-state`             | `meia/brand.py`, `meia/presets.py`, `meia/workspace.py`                       |
| 发布许可与署名   | MEIA 自有代码使用 PolyForm Noncommercial 1.0.0；商业使用需单独书面授权，第三方组件保留原许可；中英文页面统一显示 `Xiaomei_974 & codex` | `LICENSE.md`, `license/`, `app.py` |
| 公开协作边界     | 根许可证入口、Issue/PR 模板、安全报告流程与暂不接受未经邀请代码 PR 的贡献规则已经写明；CI 分别验证 Python 和三维前端 | `CONTRIBUTING.md`, `SECURITY.md`, `.github/` |

## 2. 自动化验证证据

- 2026-08-24 Python 包硬迁移验证：`meia/` 是唯一 Python 包，命令行入口为 `python -m meia.batch`；规范全称精确为 `Molecular and Extended-system Illustration Assistant`。
- `python -m compileall -q app.py meia tests`：退出 0，无语法错误。
- `python -m pytest -W error::FutureWarning -q`：438 通过，11 条来自当前 Matplotlib/Pyparsing 组合的已知弃用警告，用时 154.62 s。
- 在 `meia/components/atom_viewer/frontend` 运行 `npm test`：55/55 通过，0 失败；包含组件 iframe 重载后相机、缩放、角度步长、选择模式和选区恢复，Python 选区外部更新时仅重置选择草稿，以及 `onRender` 中状态恢复顺序不可被误删的接线契约。
- 前端 `npm run build`：Vite 6.4.2 转换 165 个主视图模块和 154 个语言组件模块，退出 0；只有压缩前 JS 大于 500 kB 的分块建议。
- `python -m pip check` 未发现依赖冲突；`npm audit --omit=dev` 报告 0 个生产依赖漏洞。当前环境未安装 `pip-audit`，因此本次没有把 Python 漏洞数据库扫描写成已完成事项。
- PolyForm 正文与官方 `1.0.0` 分支文件逐字节一致；前端锁定的 44 个非开发依赖均有对应许可归档，归档内容与 npm 包逐字节一致。Plotly 许可资源在重建后仍存在，且与上游文件共用 SHA-256 `67a26cf80f03ff388f26945dfc1f6caed1a0746ff43871190339b3aee49b94bb`。
- `python -m meia.generate_default_style` 在系统临时目录再生默认风格，`cmp` 证明与提交文件逐字节一致；临时目录已清理。
- `python -m meia.check_public_docs` 校验 13 个公开 Markdown 文件；教程中英文各 6 张 JPEG 的相对路径、大小写和文件签名均通过。
- `LICENSE.md` 与 `license/LICENSE` 逐字节一致；根入口用于 GitHub 许可证识别，辅助许可和第三方归档继续位于 `license/`。
- `python -m meia.batch --help` 退出 0；构建后 `git diff --check` 通过且 tracked `dist` 无变化。
- `python -m pytest tests/test_meia_package_namespace.py -q`：4 通过，11 条已知警告；对 tracked 路径与内容执行不区分大小写的旧品牌 token 扫描，并对旧全称大写 `S` 执行区分大小写扫描，结果均为零命中；`git diff --check` 通过。

### 当前参考产物哈希

| 产物                                              | SHA-256                                                            |
| ----------------------------------------------- | ------------------------------------------------------------------ |
| `examples/CONTCAR` | `187ee6a6d1c5bffc2b55a8ea254f0dc86c82a1f56743fcdad55504488b399d5f` |
| `examples/meia-visual-state.workspace.meia.json` | `a8288f74fc0e0a6fcc1e355ca570884ba726bb11835f019478e3803fb2910637` |
| `examples/CONTCAR_meia.svg` | `060718e8c0f4e47311a8b71d90a2831ce59150388732814b1d2fb1beef09bef8` |
| `examples/CONTCAR_meia-2.svg` | `a542c58946229aeaae1e221c54b6a73a1fc96040eb32902af24d9e443e82ba51` |
| `examples/CONTCAR_meia-2.png` | `e6fc49c28ecc650e68f1ca9447d5614df186bc2a72228b12f46f022010d734e5` |

## 3. 真实浏览器验收

从独立工作树在 8507 端口启动全新 Streamlit 进程，通过应用内浏览器加载与参考产物相同的只读 CONTCAR。验收结果：

1. 新页面完成加载后，中文标题精确为“⚛ 原子构型可视化”；切换英文后标题精确显示“MEIA - Molecular and Extended-system Illustration Assistant”；两种语言下均只显示一次统一署名 `Xiaomei_974 & codex`。
2. 上传区显示源文件名 `CONTCAR` 和 27.8 KB，应用显示“Current structure: CONTCAR (225 atoms)”；浏览器安全边界不暴露客户端任意绝对路径，页面无 Python traceback。
3. `Interactive 3D Preview` 和 `Flattened 2D Output` 均成功渲染；拖动 3D 视图后“Apply Current View”启用，滚轮缩放一次后构型可见放大，查看器仍可响应。
4. 开启选择模式后选中并确认 1 个原子；切回中文后，225 原子构型、已拖动的相机草稿、1 个原子选区、3D 视图和 2D 输出均保留。对齐 `#viewer` 的同尺寸 230×450 px 裁剪提供了缩放保留的等价数值证据：英文缩放后饱和原子着色包围盒为 99×142 px，切回中文后为 100×142 px；高度精确一致，宽度仅有 1 px 栅格差。
5. 首轮完整验收的页面 traceback/导入错误计数为 0，浏览器错误与警告日志为空，服务终端无缺失 Python 模块、locale 或 `dist` 错误。补充的 zoom-only 自动化重跑另行记录了 Plotly 对 `scene.aspectratio.x/y/z` 的 3 条 GUI-edit 警告和 1 条 `MutationObserver` 自动化警告，不用它替代首轮的干净日志结论。
6. 通过 `Ctrl+C` 停止全新进程后，8507 无监听器且服务进程已退出。

## 4. 已知边界

1. Vite 生产构建仍有大分块体积建议，本次不影响功能和验证结论。
2. 周期拓扑冲突使用保守保持原位策略；警告是可视化诊断，不等同于物理结构正确性判定。
3. 多帧轨迹和距离/角度测量仍属后续功能；多格式支持以 ASE 3.22.1 的实际解析结果为准。

## 5. 2026-08-25 大体系优化分支

`codex/large-system-optimization` 已经项目所有者真实大体系测试，并确认按实际 2D 图元数自动切换预览策略后可合并 `main`。当前已完成：

- 根据实际可见周期实例选择自动/按需 2D 预览，并防止过期图像被当作当前产物下载。
- 紧凑的运行时色彩强度表示、大体系 200 原子/页选择、稀疏 3D 选择层和 36 px 网格命中索引。
- 每个缩放帧一次 Plotly 批量更新，80 ms 最终同步；20,000 实例起在交互期临时简化键描边和氢键图层。
- 会话内单条 `RenderTopology` 缓存；仅颜色、强度、半径、键宽、相机或导出参数变化时不重新运行邻居列表和周期展开。
- 6,225 原子的临时 slab + 水工作负载完成大体系验收后已从仓库移除；日常基准改由 `python -m meia.benchmark_large_system` 在本地按需生成。

### 后端可重复基准

在 Intel macOS 26.5.2、Python 3.10.19、MEIA 0.11.0、ASE 3.22.1 和 NumPy 1.26.4 上，每组使用全新 Python 进程与独立可写 Matplotlib 缓存运行。数字是当前优化分支的后端参考值，不是浏览器 FPS，不同机器不宜直接比较绝对秒数。

| `nx` / 周期显示 | 源原子 | 显示实例 | 拓扑 / s | 3D Figure / s | 3D JSON / MiB | 2D / s | 峰值 RSS / MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 / `1×1×1` | 594 | 594 | 0.263 | 1.132 | 0.382 | 7.342 | 183.0 |
| 20 / `1×1×1` | 2,376 | 2,376 | 0.532 | 1.274 | 1.597 | 15.454 | 269.7 |
| 30 / `1×1×1` | 5,604 | 5,604 | 1.340 | 2.033 | 3.942 | 33.280 | 426.3 |
| 30 / `2×2×1` | 5,604 | 22,416 | 1.811 | 7.184 | 15.806 | 跳过 | 302.8 |

可通过 `python -m meia.benchmark_large_system --help` 调整 slab 尺寸、水层数和 a/b/c 周期范围。该基准的 2D 路径和最终导出都使用完整数据。

### 大体系真实浏览器验收

从该分支在 8512 端口启动全新 Streamlit 进程，并从空白会话直接导入基于 CONTCAR 的新大体系工作快照。验收结果：

1. 快照成功恢复 6,225 个源原子和 `1×1×1` 周期范围，页面报告 6,225 个显示实例、预计 29,200 个 2D 图元。
2. 大体系不再自动生成 2D，而是显示“生成/更新 2D 预览”和“当前会话尚未生成 2D 预览”；本次浏览器验收没有触发这项高成本手动渲染，完整 2D 路径由上表基准和自动测试覆盖。
3. 3D 视图完成加载；选择模式关闭和开启时滚轮缩放均可用，选择模式下临时选区从 3 个增加到 4 个且缩放后仍保留，查看器状态无错误。
4. 右键点击 3D 区域未出现浏览器菜单、弹窗或白屏；侧栏原子选择自动显示 32 页，每页最多 200 个原子。
5. 浏览器错误与警告日志为空；测试结束后通过 `Ctrl+C` 停止全新进程。

### 小体系合并前回归

在 8515 端口的另一个全新 Streamlit 进程中导入公开 `examples/CONTCAR`：

1. 225 个源原子正常加载，低于图元阈值时仍自动生成并更新 2D 预览。
2. 原子选择侧栏不再显示“强调当前选区为主体”及背景强度控件；其他颜色、强度、隐藏和键操作保留。
3. 先旋转视角，再开启选择、点选并确认 1 个原子，“应用当前视角”仍保持可用，未出现选择导致的相机跳变。
4. 选择模式开启和关闭时分别发送触控板滚动，3D 裁剪截图分别有 10,061 和 10,008 个字节位置变化，临时/已确认选区和相机草稿保留；浏览器日志无错误或警告。

### 分支完整验证

- `python -m compileall -q app.py meia tests` 与 `python -m meia.check_public_docs` 均退出 0；公开文档检查覆盖 13 个 Markdown 文件。
- `python -m pytest -W error::FutureWarning -q`：477 项通过，保留 11 条来自现有 Matplotlib/Pyparsing 组合的弃用警告。
- 三维前端 `npm test`：63/63 通过；`npm run build` 成功转换 168 个模块。`npm ci` 仍报告 1 项仅开发依赖的高危告警，构建仍有大分块建议；本分支没有使用强制升级处理。
