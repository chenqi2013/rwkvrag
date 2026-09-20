import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
base='http://127.0.0.1:18441'
out=Path('/home/chase/GitHub/rwkvrag/artifacts/wiki-20260919')
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True)
 page=browser.new_page(viewport={'width':1440,'height':1080});errors=[]
 page.on('pageerror',lambda error:errors.append(str(error)))
 pages=page.request.get(base+'/v1/admin/wiki').json()
 original=next(x for x in pages if x['title']=='wiki-acceptance-recovered.md')
 before=page.request.get(base+'/v1/admin/wiki/versions/'+original['id']).json()
 (out/'preview-wiki-final-v1.json').write_text(json.dumps(before,ensure_ascii=False,indent=2))
 path=Path('/tmp/rwkvrag-wiki-preview-20260919/wiki-revision-v2.md')
 path.write_text('# Wiki 验收设备修订\n\n本页为虚构验收记录。澄海 W7 的额定电压改为 24 V，冷却方式仍为自然散热。维护周期改为每 21 天一次。出厂时远程上传仍然关闭。\n')
 page.goto(base+'/admin/#/files');page.get_by_text('中文',exact=True).click()
 row=page.get_by_role('row').filter(has_text='wiki-acceptance-recovered.md')
 expect(row).to_be_visible()
 with page.expect_file_chooser() as chooser:
  row.get_by_role('button',name='上传新版本').click()
 chooser.value.set_files(str(path))
 deadline=time.monotonic()+150
 while time.monotonic()<deadline:
  items=page.request.get(base+'/v1/admin/wiki').json()
  current=next((x for x in items if x['page_id']==original['page_id']),None)
  if current and current['id']!=original['id']:break
  time.sleep(1)
 else:raise AssertionError('revision Wiki generation did not complete')
 detail=page.request.get(base+'/v1/admin/wiki/versions/'+current['id']).json()
 stale=page.request.get(base+'/v1/admin/wiki/versions/'+original['id']).json()
 assert stale['freshness']=='source_changed'
 assert detail['freshness']=='current'
 (out/'preview-wiki-final-v2.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2))
 (out/'preview-wiki-old-version-after-revision.json').write_text(json.dumps(stale,ensure_ascii=False,indent=2))
 page.goto(base+'/admin/#/wiki')
 row=page.get_by_role('row').filter(has_text='wiki-revision-v2.md');expect(row).to_be_visible()
 row.get_by_role('button',name='查看与历史').click()
 expect(page.get_by_text('原文依据',exact=True)).to_be_visible()
 page.get_by_text('[资料 1] Wiki 验收设备修订',exact=True).click()
 expect(page.get_by_text('本页为虚构验收记录。',exact=False).last).to_be_visible()
 page.screenshot(path=str(out/'wiki-current.png'),full_page=True)
 assert not errors,errors
 (out/'browser-revision-qa.json').write_text(json.dumps({'old_version':original['id'],'new_version':current['id'],
   'old_freshness':stale['freshness'],'new_freshness':detail['freshness'],'new_status':detail['status'],
   'revision_uploaded_through_browser':True,'wiki_auto_generated':True,'javascript_errors':errors},indent=2))
 print(json.dumps({'status':detail['status'],'body':detail['body'],'old_freshness':stale['freshness']},ensure_ascii=False))
 browser.close()
