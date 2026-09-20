import json,time
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

base='http://127.0.0.1:18441'
artifacts=Path('/home/chase/GitHub/rwkvrag/artifacts/wiki-20260919')
source=Path('/tmp/rwkvrag-wiki-preview-20260919/wiki-acceptance-recovered.md')
source.write_text('# Wiki 验收设备\n\n本页是功能验收使用的虚构记录。澄海 W7 的额定电压是 12 V，冷却方式是自然散热。维护周期为每 14 天一次。出厂时远程上传关闭。\n文档编号：QA-WIKI-20260919-B。\n')
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True)
 page=browser.new_page(viewport={'width':1440,'height':1000})
 errors=[];page.on('pageerror',lambda error: errors.append(str(error)))
 page.goto(base+'/admin/#/files')
 page.get_by_text('中文',exact=True).click()
 with page.expect_response(lambda r: r.url.endswith('/v1/admin/files') and r.request.method=='POST') as upload:
  page.locator('input[type=file]').first.set_input_files(str(source))
 accepted=upload.value.json();assert upload.value.status==202
 file_id=accepted['file_id']
 deadline=time.monotonic()+150
 while time.monotonic()<deadline:
  result=page.request.get(base+'/v1/admin/wiki').json()
  wiki=next((v for v in result if v['page_id']==file_id),None)
  if wiki:break
  time.sleep(2)
 else:raise AssertionError('Wiki generation timed out')
 detail=page.request.get(base+'/v1/admin/wiki/versions/'+wiki['id']).json()
 (artifacts/'preview-wiki-response.json').write_text(json.dumps(detail,ensure_ascii=False,indent=2))
 assert wiki['status']=='draft',wiki
 assert wiki['freshness']=='current'
 page.goto(base+'/admin/#/wiki')
 row=page.get_by_role('row').filter(has_text='wiki-acceptance-recovered.md')
 expect(row).to_be_visible()
 row.get_by_role('button',name='查看与历史').click()
 expect(page.get_by_text('原文依据',exact=True)).to_be_visible()
 page.get_by_text('[资料 1] Wiki 验收设备',exact=True).click()
 expect(page.get_by_text('本页是功能验收使用的虚构记录。',exact=False).last).to_be_visible()
 page.screenshot(path=str(artifacts/'wiki-preview.png'),full_page=True)
 assert not errors,errors
 (artifacts/'browser-qa.json').write_text(json.dumps({'file_id':file_id,'wiki_version':wiki['id'],'status':wiki['status'],'freshness':wiki['freshness'],'javascript_errors':errors,'uploaded_through_browser':True,'source_expansion_verified':True},indent=2))
 print(json.dumps({'file_id':file_id,'wiki_version':wiki['id'],'body':detail['body']},ensure_ascii=False))
 browser.close()
