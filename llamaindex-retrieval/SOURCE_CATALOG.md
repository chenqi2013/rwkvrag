# 来源目录与原件绑定检查

`POST /v1/admin/source-catalog/audit` 是只读诊断接口，不修复目录、不导入数据、不调用模型，也不修改答案或引用。沿用管理接口的访问边界；部署时仍须配置适当的鉴权与访问限制，本接口不新增 RBAC。

## 输入

请求对象只有以下字段：

- `sources`：1–32 个完整 `SourceItem`，直接使用问答保存的 `sources`，保留 `id`、`document_id`、`snippet` 和全部 `metadata`。
- `index_version`：可选，保存当次问答记录的索引版本。仅作为输入标记返回，不查询或证明该索引仍包含这些来源。

例如客户端已有 `answer` 和 `indexVersion` 时：

```javascript
const response = await fetch('/v1/admin/source-catalog/audit', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ sources: answer.sources, index_version: indexVersion })
});
const audit = await response.json();
```

超过32项须明确分批；不能用局部检查结果宣称全部来源已通过。不要把尚未入选的检索池作为 Writer 的证据。

## 检查与结果

逐项核对：

1. 来源显式给出的 `metadata.file_id`、`knowledge_base_id` 与 Mongo 文件记录一致；不从文档 ID、标题或 URI 猜文件 ID。
2. `source_sha256` 指向服务端文件/知识库命名空间内保留的原件，校验原件字节。不存在、损坏、越界链接分别保留为错误；不回退到最新上传文件。
3. `parsed_snapshot_sha256` 对应的解析快照完整，且绑定同一文件、知识库和原件版本。
4. `document_id` 在该解析快照中唯一，完整文档的 `source_text_sha256` 一致。
5. `source_span` 使用 Unicode code points，描述解析文档中的**索引块**；范围哈希和 `chunk_text_sha256` 均须匹配该块的保留文本。未经过 Reader 子片段选择时，该块必须逐字等于 `snippet`。
6. Reader 选中子片段时，再核对显式的 `parent_source_id`、`parent_text_sha256`、`span_start`、`span_end`、`span_sha256` 和 `offset_unit`。局部偏移必须是非布尔整数，单位必须是 `unicode_characters_in_indexed_chunk`，完整落在已验证的索引块内，且逐字等于 `snippet`。子片段 ID 必须与所给 parent ID、偏移及哈希前12位一致。任何选择字段（含 `selection_scope`）出现后都不能通过删除其他字段回退成普通整块检查；不会搜索相似/重复文本重新定位。

每项返回 `status`、原始来源身份、`snippet_sha256`、`original_verified`、`parsed_verified`、`chunk_verified`。只有全部结构绑定通过，才返回 `status: "bound"` 和 `original_url`，链接沿用 `/v1/admin/files/{file_id}/source/{source_sha256}`。

成功时还返回 `binding_scope`（`indexed_chunk` 或 `resolver_selection`）以及 `bound_document_span`（`start`、`end`、`unit: "unicode_code_points"`、`sha256`），后者是提交片段在保留解析文档中的绝对字符范围；失败时两者为 null。Reader 原有 `metadata.source_span` 仍代表索引父块，不能当作子片段范围直接使用。接口不改写这些输入元数据、模型选择、答案或引用编号，也不证明 parent ID 确实属于某个索引。

典型诊断包括 `missing_file_id`、`missing_catalog_file`、`catalog_knowledge_base_mismatch`、`missing_original_revision`、`original_integrity_error`、`missing_parsed_revision`、`parsed_identity_mismatch`、`document_text_hash_mismatch`、`snippet_span_mismatch`。目录查询本身失败时接口报错，不把数据库故障伪装成“文件不存在”。

子片段诊断另外包括 `incomplete_selection_metadata`、`invalid_selection_parent_id`、`selection_parent_hash_mismatch`、`invalid_selection_span`、`selection_snippet_mismatch`、`selection_hash_mismatch`、`selection_identity_mismatch`。仅有子片段哈希不能替代父块哈希；不隐式接受其他局部偏移协议或重建缺失的选择链。

`all_bound` 只表示本次提交的片段均通过上述核对。以下字段始终为 false：

- `index_membership_verified`：不验证当前索引成员关系、版本新鲜度或索引快照完整性。
- `context_verified`：没有在此接口核验额外的父级/表头上下文。
- `semantic_verified`：不证明解析器语义正确、资料相关、引用支持结论或答案正确。

旧原件可以在当前文件修订或删除上传路径后仍通过检查，但前提是目录记录与对应历史快照保留。历史源没有这些绑定元数据时显式报告缺口，不能生成替代哈希、改接其他知识库或伪造可访问链接。真实数据修复须另行指定目标数据库、索引版本、备份及迁移方案。

## 离线测试

```bash
uv run pytest -q tests/test_source_catalog.py tests/test_selected_source_catalog.py tests/test_source_revisions.py
```

使用临时文件、假目录及进程内 ASGI 请求；不是 Mongo/OpenSearch 集成验收，也不是浏览器或已部署服务验收。
