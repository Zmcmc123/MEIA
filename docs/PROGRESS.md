# MEIA 项目进展报告

> 日期：2026-08-24
> 阶段：v0.11.0 功能基线与公开仓库 P0 发布加固已完成本地验证

## 1. v0.11.0 当前能力

| 能力           | 当前实现                                                                                             | 主要位置                                                                          |
| ------------ | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------- |
| 双语主界面        | 支持中文与英文一键切换；中文标题保持“⚛ 原子构型可视化”，英文标题为“MEIA - Molecular and Extended-system Illustration Assistant” | `app.py`, `meia/i18n.py`, `meia/locales/`                                     |
| 语言优先级        | 手动选择高于已保存偏好，已保存偏好高于浏览器语言；手动选择通过 `meia.locale` 持久化                                                | `meia/locale_state.py`, `meia/components/locale_preference/`                  |
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
- `python -m compileall -q app.py meia scripts tests`：退出 0，无语法错误。
- `python -m pytest -W error::FutureWarning -q`：438 通过，11 条来自当前 Matplotlib/Pyparsing 组合的已知弃用警告，用时 154.62 s。
- 在 `meia/components/atom_viewer/frontend` 运行 `npm test`：55/55 通过，0 失败；包含组件 iframe 重载后相机、缩放、角度步长、选择模式和选区恢复，Python 选区外部更新时仅重置选择草稿，以及 `onRender` 中状态恢复顺序不可被误删的接线契约。
- 前端 `npm run build`：Vite 6.4.2 转换 165 个主视图模块和 154 个语言组件模块，退出 0；只有压缩前 JS 大于 500 kB 的分块建议。
- `python -m pip check` 未发现依赖冲突；`npm audit --omit=dev` 报告 0 个生产依赖漏洞。当前环境未安装 `pip-audit`，因此本次没有把 Python 漏洞数据库扫描写成已完成事项。
- PolyForm 正文与官方 `1.0.0` 分支文件逐字节一致；前端锁定的 44 个非开发依赖均有对应许可归档，归档内容与 npm 包逐字节一致。Plotly 许可资源在重建后仍存在，且与上游文件共用 SHA-256 `67a26cf80f03ff388f26945dfc1f6caed1a0746ff43871190339b3aee49b94bb`。
- `scripts/generate_default_style.py` 在系统临时目录再生默认风格，`cmp` 证明与提交文件逐字节一致；临时目录已清理。
- `scripts/check_public_docs.py` 校验 13 个公开 Markdown 文件；教程中英文各 6 张 JPEG 的相对路径、大小写和文件签名均通过。
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
