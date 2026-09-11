"""Correct observed outputs with the original stage input held byte-for-byte fixed.

Seeds are development material, never a new blind evaluation. Wrong outputs are
audit metadata only; supervised targets contain only corrected stage outputs.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TRACE = ROOT / 'llamaindex-retrieval/eval/writer-training-20260910'


def sha(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def origin(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path.read_bytes())}


def direct(phase, case):
    req = TRACE / phase / (case + '.request.json')
    res = TRACE / phase / (case + '.response.bytes')
    return read(req)['body']['contents'][0], read(res)['choices'][0]['message']['content'], [origin(req), origin(res)], None


def call(case, selector):
    path = TRACE / 'wiki-chunks-zero/completed' / (case + '.json')
    calls = read(path)['response']['generation']['model_calls']
    candidates = [c for c in calls if c['call_id'] == selector or c['stage'] == selector]
    assert len(candidates) == 1
    c = candidates[0]
    assert sha(c['prompt']) == c['prompt_sha256']
    assert sha(c['raw_text']) == c['raw_text_sha256']
    # Recheck the actual batch slot, not a reconstructed prompt.
    found = False
    for h in c['http']:
        for slot in h.get('slots', []):
            if slot['call_id'] == c['call_id']:
                assert h['payload']['contents'][slot['index']] == c['prompt']
                found = True
    assert found
    return c['prompt'], c['raw_text'], [origin(path)], c['call_id']


def main():
    rows = []


    def add(name, stage, family, source, corrected, reason, invariants):
        prompt, wrong, origins, call_id = source
        assert corrected != wrong and corrected.strip() == corrected
        assert prompt.endswith('Assistant: <think></think>' + ('' if stage == 'planner' else '\n'))
        if stage in ('planner', 'resolver'):
            json.loads(corrected)
        rows.append(dict(id='correction_' + name, stage=stage, failure_family=family,
            split='development_seed', author='root', review_status='pending_independent_review',
            source_trace=origins, call_id=call_id, original_prompt=prompt,
            prompt_sha256=sha(prompt), original_wrong_output=wrong, wrong_output_sha256=sha(wrong),
            corrected_output=corrected, target_sha256=sha(corrected), correction_reason=reason,
            generation_invariants=invariants, wrong_output_is_training_target=False,
            source_case_is_blind_evaluation=False, preserve_original_input=True))


    add('pump_citations', 'writer', 'citation_binding', direct('writer-e1', 'smoke_007'),
        'A款2022年：额定流量8 L/min，不能离线运行。[资料 1]\nA款2023年：额定流量6 L/min，支持离线运行。[资料 1]\nB款2023年：额定质量流量4 kg/min，不能离线运行。[资料 2]',
        'A款2023与B款2023引用错绑；保留正确事实，逐事实绑定真正来源。',
        ['同对象不同年份与不同对象同时出现', '来源顺序和问题顺序独立变化，包括首答案引用1', '单位不换算'])
    add('reef_dates', 'writer', 'entity_field_binding', call('held_021', 'writer'),
        '红山毛榉号：2000年6月10日置放为人工鱼礁。[资料 1]\n红橡号：1999年9月13日置放为人工鱼礁。[资料 1]',
        '完整表格已可见；按结局筛选两行，绑定置放日期，不取退役或下水日期。',
        ['保留完整行列和竞争日期', '匹配行数与行顺序变化', '目标和干扰日期年份交叠，不能凭年份筛选'])
    add('release_reference_date', 'writer', 'entity_field_binding', direct('writer-zero', 'writer_v6_eval_3667615_01_all_units'),
        '《殺手少女》真人电影于2015年4月11日在日本上映。[资料 2]',
        '资料3的2015-04-03是参考文献日期；上映事实在资料2。',
        ['同一对象同时有事件日期与参考文献日期', '保留标题和事件语义', '随机化日期先后及引用位置'])
    add('can_units', 'writer', 'units_and_qualifiers', call('held_004', 'writer'),
        '消息ID支持11位和29位。[资料 1]\n每帧最大实际数据量为64字节（byte）。[资料 2]\n条目给出的最高资料率为5 Mbit/s。[资料 2]\n可达的最高速率受电气条件和组态影响，包括节点个数、总线长度及其他电磁因素。[资料 1]',
        '64字节不能写成64位；速率和线路条件分别取支持证据。',
        ['位与字节出现在同一主题', '保留速率适用条件', '禁止跨字段挪用数值'])
    add('fee_scope', 'writer', 'units_and_qualifiers', direct('writer-zero', 'writer_v6_eval_6066934_02_all_units'),
        '卢伟国建议豁免自修生一半考试费，认为这已足够杜绝捣乱，不应一刀切将自修生从计划中剔除。[资料 3]',
        '原输出遗漏一半和不应一刀切剔除的限定，不能改成全额豁免。',
        ['保留分数、否定及适用对象', '同主题对照全额与部分', '建议不能写成已执行政策'])
    add('missing_score', 'writer', 'missing_evidence', direct('writer-zero', 'writer_v6_eval_370260_03_all_units'),
        '所给资料未提供仲满男子个人佩剑决赛的最终比分，无法据此确定。',
        '奖牌与赛事资料不能支持决赛比分，删去编造比分与引用。',
        ['保留相关但不含所求事实的资料', '缺失判断覆盖全部可见证据'])
    add('planned_not_actual', 'writer', 'missing_evidence', direct('writer-e1', 'writer_v6_eval_6066934_03_all_units'),
        '所给资料记载的是计划派发一万张门票。[资料 1]\n未提供最后实际使用的张数，无法据此确定实际使用量。',
        '计划派发数量不等于实际使用数量。',
        ['保留诱人的计划数', '目标明确区分计划与实绩', '增加实际数可见的正例对照'])
    add('aggregate_not_each', 'writer', 'missing_evidence', direct('writer-e1', 'writer_v6_eval_6066934_04_all_units'),
        '资料只说明预留20亿元在未来十年推行街市现代化计划。[资料 2]\n未提供每个街市分别获拨的金额，无法据此列出各街市资金。',
        '整体计划预算不能当作每个街市的拨款。',
        ['总额与每对象金额区分', '不得平均分摊推算未给出的分项'])
    add('empty', 'writer', 'missing_evidence', direct('writer-zero', 'writer_v6_eval_370260_01_empty_evidence'),
        '当前没有可用证据，无法据此确定所问日期和场馆。',
        '空证据不能生成日期、场馆或虚构资料编号。',
        ['真正空证据', '只说明当前依据不足，不声称现实不存在'])
    add('train_multipart', 'writer', 'multi_part_coverage', direct('writer-e1', 'smoke_002'),
        'ICE-1不能与其他列车组合运行，只能单独运行。[资料 1]\nICE-2科隆／杜塞尔多夫往柏林的例子在Hamm站合体。[资料 5]\nJR九州305系每编组共6辆，其中4辆为动力车。[资料 8]',
        '三个问题分别作答，不能用重联方式替代编组总数和动力车数。',
        ['多对象多字段同时保留', '每一子问题都要覆盖', '相近概念证据作为干扰'])
    add('history_active', 'writer', 'history_correction', direct('writer-zero', 'smoke_004'),
        'Windows NT 3.5工作站版允许10个客户端并发访问文件服务器。[资料 1]\n不支持Mac客户端。[资料 1]\n开发代号为Daytona。[资料 1]',
        '按完整历史保留最终三格，不恢复3.51、Server版或已删除协议背景。',
        ['保留真实长历史中的分散更正', '最新问题依赖指代', '保留要求与撤回要求交错，不能只依赖全部撤回模板'])
    add('stop_after_answer', 'writer', 'bounded_answer_eos', direct('writer-zero', 'writer_v6_eval_4271857_02_all_units'),
        '乌鲁伊在昆士兰州艾尔利海滩附近登陆。[资料 3]',
        '首个结论已经正确，删除循环复述，完整答案后训练EOS。',
        ['保留大段干扰证据', '短目标覆盖完整有效问题', 'EOS进入监督标签'])
    for name, selector, reason in [
        ('reader_books', 'd68973a0-b809-4e19-b983-69839624782e', '该单元明确给出绿皮书、黄皮书、蓝皮书，应YES。'),
        ('reader_stages', '23c0140c-df3e-4821-b98d-64507bd4a401', '该单元明确给出安全生命周期16阶段，应YES。')]:
        add(name, 'resolver', 'reader_false_negative', call('held_010', selector), '{"answer": "YES"}', reason,
            ['原问与原单元保持一致', '保留完整父级上下文', '长正文中的直接答案也应选中'])
    add('reader_relative', 'resolver', 'reader_relative_absolute', call('held_004', '3ba1aa83-f1e7-4fb9-bcc0-362a659bc520'),
        '{"answer": "NO"}', '该单元仅提供传统CAN的5倍，未提供绝对Mbit/s或基数；不能借用另一个单元。',
        ['相对数与绝对数对照', '单元可见性边界', '补充绝对数或改问比例时应YES'])
    add('writer_partial_after_reader', 'writer', 'missing_evidence', call('held_010', 'writer'),
        'IEC 61131-3条目列出的五种语言是指令表（ILD）、阶梯图（LD）、功能区块图（FBD）、结构化文字（STL）和顺序功能流程图（SFC）。[资料 2]\nIEC 61499以事件驱动模型取代循环执行模型。[资料 1]\n当前所给证据未提供DLMS三份规范文档的书籍颜色，也未提供IEC 61508安全生命周期的阶段数，无法据此确定这两项。',
        'Writer实际仅看到两份证据；Reader另两份虽有答案，但没有进入这个调用，不能倒灌为Writer目标。',
        ['部分问题有证据、部分没有', '严格按当前阶段可见证据纠错', '禁止用完整语料替代当前输入边界'])
    add('planner_history', 'planner', 'planner_contract', call('held_019', 'planner'),
        json.dumps({'queries': ['Windows NT 3.5 工作站版 文件服务器 并发客户端上限 Mac客户端支持', 'Windows NT 3.5 开发代号'],
            'fields': ['Windows NT 3.5工作站版允许的并发文件访问客户端数', 'Windows NT 3.5工作站版是否支持Mac客户端', 'Windows NT 3.5开发代号']}, ensure_ascii=False),
        '原输出7条queries越界且复制历史；压缩为涵盖当前有效要求的两条检索式，不填答案。',
        ['输出1至6条必要且不同检索式', '有效要求不能因压缩丢失', '撤回对象不进入检索'])

    out = HERE / 'correction-seeds-v1'
    out.mkdir(exist_ok=False)
    path = out / 'seeds.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    (out / 'MANIFEST.json').write_text(json.dumps(dict(schema='rwkv_corrected_trace_seeds_v1', count=len(rows),
        seed_file_sha256=sha(path.read_bytes()), status='pending_independent_review',
        author='root', generation_order=['observed original input + wrong output', 'evidence-grounded correction',
            'independent correction review', 'new-source variants preserving causal structure', 'independent row review'],
        training_target_policy='corrected output only; wrong output retained only in audit metadata',
        evaluation_policy='These exposed source cases and derivatives are development regressions; retain fresh disjoint sources for blind evaluation.',
        existing_draft_v4_status='2000 labels reviewed, but not yet admitted as descendants of corrected seeds; do not relabel retroactively.'), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'seeds': len(rows), 'sha256': sha(path.read_bytes())}))


if __name__ == "__main__":
    main()
