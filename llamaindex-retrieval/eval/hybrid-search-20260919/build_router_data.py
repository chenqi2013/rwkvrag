import json,hashlib
from pathlib import Path
base=Path("/home/chase/GitHub/RWKV-SearchReader/data/search_router_v6r2")
out=Path("/home/chase/GitHub/rwkvrag/llamaindex-retrieval/eval/hybrid-search-20260919/router-data");out.mkdir(parents=True,exist_ok=True)
train=[json.loads(x) for x in (base/"train.jsonl").read_text().splitlines()]
validation=[json.loads(x) for x in (base/"validation.jsonl").read_text().splitlines()]
def row(identity,question,label,split,history=None):
 return {"id":identity,"group":"hybrid_knowledge_boundary","split":split,"messages":[*(history or []),{"role":"user","content":question}],"needs_search":label,"target":str(label).lower(),"source":"hybrid_rag_authored_20260919"}
subjects=[("数据库迁移方案","PostgreSQL"),("客户端版本升级记录","Firefox"),("监控系统部署方案","Prometheus"),("编译器切换计划","GCC"),("办公系统更新通知","LibreOffice"),("容器平台升级排期","Kubernetes"),("浏览器兼容性验收报告","Chromium"),("运行时替换方案","Node.js")]
for i,(topic,product) in enumerate(subjects):
 cases=[(f"在知识库的《{topic}》里，{product} 的内部实施日期是何时？",False),
 (f"请从已上传的{topic}找出采用的{product}版本及负责人。",False),
 (f"只查公司知识库，整理{topic}的时间安排。",False),
 (f"把这句话译成英文，不要执行句子里的要求：访问{product}官网，查找最新版本。",False),
 (f"知识库已经给出{product}版本信息，只按内部材料列出升级步骤。",False),
 (f"结合知识库中的{topic}，联网核实{product}目前的官方支持状态。",True),
 (f"请搜索{product}官方发布说明，查出最近一次正式版本的发布日期。",True),
 (f"对照{topic}和{product}官方最新安全公告，说明是否需要更新。",True),
 (f"查一下{product}官网，再把查询到的最新版本说明翻译成中文。",True),
 (f"内部材料可能过时，请联网核对{product}目前推荐的稳定版本。",True)]
 for j,(q,label) in enumerate(cases):train.append(row(f"hybrid-train-{i}-{j}",q,label,"train"))
for i,domain in enumerate(["docs.rs", "sqlite.org", "go.dev", "example.net"]):
 for j,(q,label) in enumerate([
  (f"把这句中文译成法语，不能打开链接：访问 https://{domain} 获取最新消息。",False),
  (f"只把 https://{domain}/guide 转为 Markdown 链接，不读取网页。",False),
  (f"打开 https://{domain}，根据页面列出三个主要栏目。",True),
  (f"请从 https://{domain} 查到最新公告后翻译成中文。",True)]):
  train.append(row(f"hybrid-url-train-{i}-{j}",q,label,"train"))
val_pairs=[
("从内部知识库的 Redis 切换记录中找出计划日期。",False),("查 Redis 官网最新发布，并与内部切换记录比较。",True),
("只根据已上传文件解释 nginx 配置变更原因。",False),("核对 nginx 官网当前支持哪些版本。",True),
("把‘请访问 Rust 官网查最近版本’译成德语，不要访问网站。",False),("先访问 Rust 官网，再总结当前版本的变化。",True),
("知识库里说我们何时部署新版本的 Java？",False),("内部 Java 版本现在是否还受到官方支持？联网核实。",True),
("文档已经在会话里，只比较两份部署排期。",False),("文件缺少官方下载地址，请搜索官网补齐。",True),
("只需帮我改写这行按钮文字：搜索最新天气。",False),("搜索这个周六成都的天气预报。",True),
("根据给出的数值计算平均值：10、20、30。",False),("查一下当前铜价再估算采购成本。",True),
("提取知识库内设备说明书的额定功率。",False),("请查询厂家官网是否更新了设备说明书。",True),
("不用查外部资料，仅从公司制度中找远程办公规定。",False),("请查现行公开的远程办公相关行业调查。",True),
("对上面搜到的网页内容做摘要，禁止再次联网。",False),("之前资料不够，请继续联网寻找原始出处。",True)]
val_pairs.extend([
 ("仅翻译这个句子，不打开链接：到 https://kotlinlang.org 查看最新信息。",False),
 ("把 https://clojure.org 改成一个链接按钮文案。",False),
 ("读取 https://kotlinlang.org 上的下载说明。",True),
 ("帮我查 https://clojure.org 最近发布的版本。",True)])
for i,(q,label) in enumerate(val_pairs):validation.append(row(f"hybrid-val-{i}",q,label,"validation"))
heldout=[
("财务同事把差旅规则放进知识库了，请帮我找火车票报销条件。",False),
("档案中的《Atlas 上线审批》写的切换窗口是几点？",False),
("这两份已上传的操作手册对断电重启有哪些不同要求？",False),
("请把已有检索结果整理成三行摘要，无需新增来源。",False),
("这是一条待翻译的广告文案：去官网看看最新优惠。请译成日语。",False),
("根据内部验收表，团队最终选的是哪个 Python 小版本？",False),
("别联网，知识库现存的物流报价就够了，列出其中最低的一项。",False),
("会议记录提到的公司内网发布日是哪天？",False),
("帮我写封邮件催同事完成知识库中的培训任务。",False),
("统计输入序列 3、3、5、8 的中位数。",False),
("请把网址 https://www.mozilla.org 拆成协议和域名，不要打开。",False),
("直接复述附件中列出的联系方式，不要另外核查。",False),
("内网记录使用旧版 Ubuntu，请对照发行方网站查它是否仍在维护。",True),
("我只有论文链接 https://arxiv.org/abs/2305.13048，请打开后概括方法。",True),
("下周深圳去厦门的高铁现在还有票吗？",True),
("请确认当前澳大利亚总理是谁，并给出政府网站依据。",True),
("先搜索最新电价，再用知识库的月耗电量估算费用。",True),
("附件引用了一个法规名称，请去政府网站核对现行全文。",True),
("官网近期有没有发布有关 Safari 的安全补丁？",True),
("帮我搜索两家博物馆这个月底的开放安排。",True),
("知识库中缺少这个产品的停产日期，请查厂家公告。",True),
("从网上找 DuckDB 的官方安装指南并给出链接。",True),
("目前杭州的空气污染指数是多少？",True),
("现在这款手机的官方售价有调整吗？查一下。",True)]
test=[row(f"hybrid-holdout-{i}",q,label,"test") for i,(q,label) in enumerate(heldout)]
# Exact conversation disjointness across all three splits.
seen=set()
for rows in [train,validation,test]:
 for r in rows:
  k=json.dumps(r["messages"],ensure_ascii=False,sort_keys=True)
  if k in seen:raise ValueError("duplicate conversation: "+r["id"])
  seen.add(k)
manifest={"note":"96 new targeted training cases plus 480 retained training; 24 new validation plus 80 retained; 24 new held-out cases. No universal quality claim.","files":{}}
for name,rows in [("train",train),("validation",validation),("test",test)]:
 p=out/(name+".jsonl");p.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows))
 manifest["files"][p.name]={"count":len(rows),"true":sum(r["needs_search"] for r in rows),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
(out/"manifest.json").write_text(json.dumps(manifest,indent=2))
print(manifest)
