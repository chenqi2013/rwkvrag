import test from 'node:test';
import assert from 'node:assert/strict';
import {evidenceFlowNotices, executionLabel} from '../src/evidenceFlow.ts';
import {evidenceWarnings} from '../src/answerPresentation.ts';
const base = {protocol:'evidence-flow-v1', failed_node_count:0, failed_cell_count:0,
  unexamined_job_count:0, verified_unselected_fact_ids:[]};

test('materials lost in processing are not described as absent from sources', () => {
  const response={answer:'原文没有',sources:[],generation:{evidence_flow:{...base,state:'no_selected_evidence',input_source_count:2}}};
  const original=JSON.stringify(response);
  const notes=evidenceWarnings(response);
  assert.equal(notes.length,1);
  assert.match(notes[0][0],/2 份材料进入处理/);
  assert.match(notes[0][0],/不能据此断言原文未记载/);
  assert.equal(JSON.stringify(response),original);
});
test('empty input and missing trace remain distinct', () => {
  assert.match(evidenceFlowNotices({...base,state:'no_input_materials'})[0][0],/没有材料进入/);
  assert.match(evidenceFlowNotices({...base,state:'unrecorded'})[0][0],/记录缺失/);
  assert.deepEqual(evidenceFlowNotices(),[]);
});
test('existing final evidence does not hide unexamined work or omitted verified records', () => {
  const notes=evidenceFlowNotices({...base,state:'selected_evidence_present',unexamined_job_count:3,
    failed_node_count:1,failed_cell_count:1,verified_unselected_fact_ids:['A2']});
  assert.equal(notes.length,3);
  assert.match(notes[1][0],/3 项抽取任务未检查/);
  assert.match(notes[2][0],/未被字段归并层选中/);
});
test('completed execution never displays a semantic success label', () => {
  assert.equal(executionLabel('completed'),'执行完成 · 不代表回答正确');
  assert.match(executionLabel('length'),/回答未完成/);
  assert.match(executionLabel('funnel_partial_failure'),/部分处理失败/);
});

test('unknown diagnostic protocols retain the existing no-evidence warning', () => {
  const notes=evidenceWarnings({sources:[],generation:{evidence_flow:{protocol:'future-v99'}}});
  assert.equal(notes.length,1);
  assert.match(notes[0][0],/未返回有效证据/);
});
