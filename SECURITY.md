# Security Policy / 安全策略

## Supported version / 支持版本

Security fixes are applied to the latest code on `main` until MEIA publishes a
stable release series. Older snapshots and downloaded ZIP files are not
maintained separately.

在 MEIA 发布稳定版本系列前，安全修复只应用于 `main` 的最新代码；旧快照和已下载的 ZIP
不单独维护。

## Private reporting / 私下报告

Do not disclose a suspected vulnerability in a public issue. Use **Security →
Report a vulnerability** on the GitHub repository when that option is
available. If it is unavailable, contact the repository owner through the
channel published on the owner's GitHub profile and include "MEIA security" in
the subject.

请勿在公开 Issue 中披露疑似漏洞。优先使用 GitHub 仓库的 **Security → Report a
vulnerability** 私密报告入口；若该入口不可用，请通过仓库所有者 GitHub 个人资料中公开的
联系方式联系，并在主题中注明“MEIA security”。

Include the affected commit or version, reproduction steps, impact, and any
suggested mitigation. Do not send secrets, personal data, or confidential
atomic structures unless the maintainer explicitly requests a safe transfer
method.

报告中请包含受影响的提交或版本、复现步骤、影响和可行的缓解建议。除非维护者另行提供安全
传输方式，否则不要发送密钥、个人数据或机密构型。
