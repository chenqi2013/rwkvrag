# 模型大小配对：2026-09-21

188条成员、180个不同prompt；两模型各两轮752次，0运行错误。已见固定材料开发回归，两模型均零State；不是生产检索或独立质量验收。

- [结果与口径](../../docs/archive/2026-09/model-size-paired-results-20260921.md)
- [逐题审读](review1/REVIEW.md)：原始回答SHA绑定，由实现者审读，非独立评审。
- [汇总](review1/SUMMARY.json)、[完整判读](review1/REVIEW.json)
- [原始归档](raw-run2.tar.gz)：1519文件，请求、响应、token、日志与运行绑定。
- [文件清单](RAW-MANIFEST.json)、[完整性核验](INTEGRITY.json)
- [转换核验](CONVERSION.json)、[依赖追加核对](DEPENDENCIES-AUDIT.json)、[清理](CLEANUP.json)
- [首版预检停止记录](RUN1-PREFLIGHT-ABORT.json)：0次生成，旧文件未改。
- [审读源码绑定](REVIEW-BINDING.json)；review1保留初稿快照及一致性更正，未修补原始答案。

两轮事实完整正确2.9B均99/188、7.2B均121/188；持续复读分别29和15。严格通过75对87/88。7.2B五项原文跨轮变化，根因未知。

## 前端回放

[浏览器检查](frontend/BROWSER-CHECK.json)、[展示源码与数据绑定](frontend/BINDING.json)。页面为 `/admin/#/model-comparison`。展示数据在 `llamaindex-retrieval/web/public/experiments/model-size-paired-20260921.json`；可用实验目录下 `export_frontend.py ALL-ANSWERS.json REVIEW.json OUTPUT.json` 重建，原文SHA必须匹配。
