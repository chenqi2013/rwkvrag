import { ArrowUpOutlined, DeleteOutlined, DownloadOutlined, MenuOutlined, MessageOutlined, PlusOutlined, SettingOutlined, StopOutlined } from "@ant-design/icons";
import { Alert, Button, Drawer, Input, Modal, Select, Spin } from "antd";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { useLanguage } from "../i18n";
import type { KnowledgeBase } from "../types";
import { errorMessage } from "../utils";
import ChatResponse from "../components/ChatResponse";
import { carriedAnswer, chatRequest, conversationHistory, interruptPending, isAskResponse, newChatId, newConversation, requestIssue, reviseConversation, settleTurn, type ChatScope, type Conversation } from "../chatState";
import { loadConversations, persistConversation } from "../chatStorage";
import "../chat.css";

type Pending = { conversationId: string; turnId: string; controller: AbortController };
const ACTIVE_KEY = "rwkvrag-chat-active-v1";
const titleOf = (c: Conversation) => c.turns[0]?.question.trim() || "";
const id = newChatId;

export default function ChatPage() {
  const { tr, language, setLanguage } = useLanguage();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const conversationsRef = useRef(conversations);
  const [activeId, setActiveId] = useState("");
  const activeRef = useRef("");
  const [draft, setDraft] = useState("");
  const draftRef = useRef("");
  const [ready, setReady] = useState(false);
  const [busy, setBusy] = useState<Pending>();
  const pending = useRef<Pending | undefined>(undefined);
  const mounted = useRef(true);
  const writeQueue = useRef(Promise.resolve());
  const storageFailed = useRef(false);
  const [storageError, setStorageError] = useState("");
  const [unreadable, setUnreadable] = useState(0);
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [kbError, setKbError] = useState("");
  const [mobileOpen, setMobileOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [deletingId, setDeletingId] = useState<string>();
  const [filter, setFilter] = useState("");
  const scroll = useRef<HTMLDivElement>(null);
  const followBottom = useRef(true);
  const [below, setBelow] = useState(false);
  const input = useRef<React.ComponentRef<typeof Input.TextArea>>(null);

  function queueSave(c: Conversation, deleting = false) {
    writeQueue.current = writeQueue.current.then(async () => {
      if (storageFailed.current) return;
      await persistConversation(c, deleting);
    }).catch(error => {
      storageFailed.current = true;
      if (mounted.current) setStorageError(errorMessage(error));
    });
  }
  function publish(rows: Conversation[]) {
    conversationsRef.current = rows;
    if (mounted.current) setConversations(rows);
  }
  function update(c: Conversation) {
    publish(conversationsRef.current.map(row => row.id === c.id ? c : row));
    queueSave(c);
  }
  function saveDraft() {
    const c = conversationsRef.current.find(row => row.id === activeRef.current);
    if (c && c.draft !== draftRef.current) update(reviseConversation(c, { draft: draftRef.current }));
  }
  function activate(c: Conversation) {
    activeRef.current = c.id;
    setActiveId(c.id);
    draftRef.current = c.draft;
    setDraft(c.draft);
    try { sessionStorage.setItem(ACTIVE_KEY, c.id); } catch { /* Only selection, not conversation data. */ }
    followBottom.current = true;
    setBelow(false);
    setMobileOpen(false);
    requestAnimationFrame(() => input.current?.focus());
  }
  function create() {
    saveDraft();
    const c = newConversation(id());
    publish([c, ...conversationsRef.current]);
    queueSave(c);
    activate(c);
  }
  function stop(status: "cancelled" | "interrupted" = "cancelled") {
    const owner = pending.current;
    if (!owner) return;
    pending.current = undefined; // Invalidate before abort or any later resolution.
    owner.controller.abort();
    const c = conversationsRef.current.find(row => row.id === owner.conversationId);
    if (c) update(settleTurn(c, owner.turnId, { status }));
    if (mounted.current) setBusy(undefined);
  }

  useEffect(() => {
    mounted.current = true;
    let alive = true;
    void loadConversations().then(({ conversations: rows, unreadable: count }) => {
      if (!alive) return;
      setUnreadable(count);
      const restored = rows.map(c => {
        const next = interruptPending(c);
        if (next !== c) queueSave(next);
        return next;
      }).sort((a, b) => b.updatedAt - a.updatedAt);
      publish(restored);
      let remembered = "";
      try { remembered = sessionStorage.getItem(ACTIVE_KEY) || ""; } catch { /* Optional selection. */ }
      if (restored.length) activate(restored.find(c => c.id === remembered) ?? restored[0]);
      else create();
      setReady(true);
    }).catch(error => {
      if (!alive) return;
      storageFailed.current = true;
      setStorageError(errorMessage(error));
      create();
      setReady(true);
    });
    void api.knowledgeBases().then(rows => { if (alive) setKnowledgeBases(rows); })
      .catch(error => { if (alive) setKbError(errorMessage(error)); });
    const unload = () => { saveDraft(); stop("interrupted"); };
    window.addEventListener("pagehide", unload);
    return () => {
      alive = false;
      mounted.current = false;
      window.removeEventListener("pagehide", unload);
      saveDraft();
      stop("interrupted");
    };
    // All lifecycle mutation functions read current refs, not render snapshots.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const active = conversations.find(c => c.id === activeId);
  useEffect(() => {
    const node = scroll.current;
    if (!node) return;
    if (followBottom.current) node.scrollTop = node.scrollHeight;
    else setBelow(true);
  }, [activeId, active?.turns]);

  async function send() {
    if (!ready || pending.current) return; // Synchronous guard also blocks double Enter.
    const current = conversationsRef.current.find(c => c.id === activeRef.current);
    if (!current) return;
    const withDraft = { ...current, draft: draftRef.current };
    if (requestIssue(withDraft)) return;
    const payload = chatRequest(withDraft);
    const owner: Pending = { conversationId: current.id, turnId: id(), controller: new AbortController() };
    pending.current = owner;
    setBusy(owner);
    update(reviseConversation(current, { draft: "", turns: [...current.turns,
      { id: owner.turnId, question: payload.question, status: "pending", startedAt: Date.now() }] }));
    draftRef.current = "";
    setDraft("");
    followBottom.current = true;
    setBelow(false);
    try {
      const response = await api.ask(payload, owner.controller.signal);
      if (pending.current !== owner || !mounted.current) return;
      const c = conversationsRef.current.find(row => row.id === owner.conversationId);
      if (c) update(settleTurn(c, owner.turnId, isAskResponse(response)
        ? { status: "received", response }
        : { status: "failed", invalidResponse: response,
          error: tr("响应结构无效；原响应保留在下方。", "Invalid response envelope; the original response is retained below.") }));
    } catch (error) {
      if (pending.current !== owner || !mounted.current) return;
      const c = conversationsRef.current.find(row => row.id === owner.conversationId);
      if (c) update(settleTurn(c, owner.turnId, { status: "failed", error: errorMessage(error) }));
    } finally {
      if (pending.current === owner) {
        pending.current = undefined;
        if (mounted.current) setBusy(undefined);
      }
    }
  }

  function removeConversation() {
    const c = conversationsRef.current.find(row => row.id === deletingId);
    if (!c) { setDeletingId(undefined); return; }
    if (pending.current?.conversationId === c.id) stop();
    const latest = conversationsRef.current.find(row => row.id === c.id)!;
    queueSave(latest, true);
    const remaining = conversationsRef.current.filter(row => row.id !== c.id);
    publish(remaining);
    if (activeRef.current === c.id) {
      activeRef.current = ""; // Do not save a deleted draft during create().
      if (remaining.length) activate(remaining[0]);
      else create();
    }
    setDeletingId(undefined);
  }

  function exportConversation() {
    saveDraft();
    const c = conversationsRef.current.find(row => row.id === activeRef.current);
    if (!c) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(c, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `rwkvrag-chat-${c.id}.json`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const modes = [
    { value: "knowledge_base", label: tr("仅知识库", "Knowledge base") },
    { value: "web", label: tr("仅网络", "Web") },
    { value: "hybrid", label: tr("知识库 + 网络", "KB + web") },
    { value: "auto", label: tr("自动检索", "Auto retrieval") },
  ];
  const scopeLabel = modes.find(m => m.value === active?.scope.retrieval_mode)?.label;
  const contextLength = active ? conversationHistory(active).length : 0;
  const issue = active ? requestIssue({ ...active, draft }) : "empty";
  const issueText = issue === "question_limit" ? tr("问题超过 10000 字符；请自行缩短。", "Question exceeds 10,000 characters; please shorten it.")
    : issue === "history_limit" ? tr("已达 32 轮历史上限，请新建对话；不会静默截断历史。", "32-turn history limit reached. Start a new chat; history is never silently truncated.")
      : issue === "message_limit" ? tr("历史消息超过接口长度限制，请新建对话。", "A history message exceeds the API limit. Start a new chat.") : "";
  function changeScope(changes: Partial<ChatScope>) {
    const c = conversationsRef.current.find(row => row.id === activeRef.current);
    if (c && !c.turns.length) update(reviseConversation(c, { scope: { ...c.scope, ...changes } }));
  }
  const sidebar = <>
    <div className="chat-brand"><span className="chat-mark">R</span><strong>RWKV<span> · RAG</span></strong></div>
    <Button className="chat-new" icon={<PlusOutlined />} onClick={create} disabled={!ready}>{tr("新建对话", "New chat")}</Button>
    <Input className="chat-search" allowClear value={filter} onChange={e => setFilter(e.target.value)}
      placeholder={tr("搜索对话", "Search chats")} aria-label={tr("搜索对话", "Search chats")} />
    <div className="chat-history-heading">{tr("最近对话", "Recent chats")}</div>
    <nav className="chat-history" aria-label={tr("对话列表", "Conversations")}>
      {[...conversations].sort((a, b) => b.updatedAt - a.updatedAt)
        .filter(c => (titleOf(c) || tr("新对话", "New chat")).toLocaleLowerCase().includes(filter.toLocaleLowerCase()))
        .map(c => <div className={`chat-history-row${c.id === activeId ? " active" : ""}`} key={c.id}>
          <button type="button" className="chat-history-select" aria-current={c.id === activeId ? "page" : undefined}
            title={titleOf(c) || tr("新对话", "New chat")} onClick={() => { saveDraft(); activate(conversationsRef.current.find(row => row.id === c.id)!); }}>
            <MessageOutlined /><span>{titleOf(c) || tr("新对话", "New chat")}</span>
            {busy?.conversationId === c.id && <Spin size="small" />}
          </button>
          <Button type="text" size="small" className="chat-delete" icon={<DeleteOutlined />}
            aria-label={tr("删除对话", "Delete chat")} onClick={() => setDeletingId(c.id)} />
        </div>)}
    </nav>
    <footer className="chat-sidebar-footer">
      <Link to="/dashboard"><SettingOutlined /> {tr("知识库与管理后台", "Knowledge console")}</Link>
      <span>{tr("会话列表保存在此浏览器，不跨设备同步。", "Chats are stored in this browser, not synced across devices.")}</span>
    </footer>
  </>;

  return <div className="chat-shell">
    <aside className="chat-sidebar">{sidebar}</aside>
    <Drawer className="chat-mobile-drawer" placement="left" width={280} open={mobileOpen} onClose={() => setMobileOpen(false)}
      title={tr("我的对话", "My chats")}><div className="chat-sidebar-content">{sidebar}</div></Drawer>
    <main className={`chat-main${active?.turns.length ? "" : " is-empty"}`}>
      <header className="chat-topbar">
        <Button className="chat-menu" type="text" icon={<MenuOutlined />} onClick={() => setMobileOpen(true)}
          aria-label={tr("打开会话列表", "Open conversations")} />
        <div className="chat-topbar-title">{active && titleOf(active) || tr("新的对话", "New conversation")}</div>
        <Button type="text" className="chat-language" onClick={() => setLanguage(language === "zh" ? "en" : "zh")}
          aria-label={tr("切换语言", "Switch language")}>{language === "zh" ? "EN" : "中文"}</Button>
        <Button type="text" icon={<DownloadOutlined />} onClick={exportConversation} disabled={!ready}
          aria-label={tr("导出当前对话（含原始记录）", "Export chat including raw records")} />
      </header>
      {storageError && <Alert className="chat-banner" type="error" showIcon title={tr("未能保存：本页后续更改仅在内存中", "Not saved: further changes are in memory only")}
        description={<>{tr("请先导出再刷新。已有存储不会被覆盖；删除操作也可能未保存。", "Export before reloading. Existing storage was not overwritten; deletions may also be unsaved.")} {storageError}</>} />}
      {unreadable > 0 && <Alert className="chat-banner" type="warning" title={tr(`${unreadable} 条会话无法读取，原存储保留且未覆盖。`, `${unreadable} unreadable chats remain in storage, untouched.`)} />}
      <div className="chat-messages" ref={scroll} onScroll={e => {
        const n = e.currentTarget;
        followBottom.current = n.scrollHeight - n.scrollTop - n.clientHeight < 100;
        if (followBottom.current) setBelow(false);
      }}>
        <div className="chat-conversation">
          {!ready ? <div className="chat-welcome"><Spin /></div> : !active?.turns.length ? <section className="chat-welcome">
            <div className="chat-welcome-logo"><span className="chat-mark">R</span><strong>RWKV</strong></div>
            <h1>{tr("有什么可以帮你？", "How can I help you?")}</h1>
            <p>{tr("从资料出发，继续每一次追问。", "Start with sources. Keep the conversation going.")}</p>
          </section> : active.turns.map(turn => <section className="chat-turn" key={turn.id}>
            <div className="chat-user" aria-label={tr("你", "You")}><div className="chat-user-bubble">{turn.question}</div></div>
            <div className="chat-assistant"><span className="chat-mark" aria-hidden="true">R</span><div className="chat-assistant-body">
              {turn.status === "pending" ? <div className="chat-pending" role="status"><Spin size="small" /> {tr("正在等待服务返回…", "Waiting for the service…")}</div>
                : turn.response ? <ChatResponse response={turn.response} />
                  : <Alert type={turn.status === "failed" ? "error" : "info"} showIcon title={turn.status === "failed"
                    ? tr("请求失败，没有用其他答案替换", "Request failed; no substitute answer")
                    : turn.status === "interrupted" ? tr("等待已中断，未自动重发", "Waiting interrupted; not automatically resent")
                      : tr("已停止等待", "Stopped waiting")}
                    description={turn.error || tr("未收到完整响应；服务端是否停止需另行核实。", "No complete response received; server cancellation is not confirmed.")} />}
              {turn.invalidResponse !== undefined && <details className="answer-trace"><summary>{tr("异常响应原文", "Invalid response envelope")}</summary>
                <pre className="trace-json">{JSON.stringify(turn.invalidResponse, null, 2)}</pre>
              </details>}
              {turn.status !== "pending" && carriedAnswer(turn) === undefined && <p className="chat-history-notice">
                {tr("此轮未加入下一次请求的历史；失败或未完成输出仍保留在本页。", "This turn is excluded from the next request's history; failed or incomplete output remains visible here.")}
              </p>}
            </div></div>
          </section>)}
        </div>
      </div>
      <div className="chat-composer-wrap">
        {below && <Button className="chat-bottom" onClick={() => { followBottom.current = true; if (scroll.current) scroll.current.scrollTop = scroll.current.scrollHeight; setBelow(false); }}>{tr("回到最新消息 ↓", "Latest message ↓")}</Button>}
        <form className="chat-composer" onSubmit={e => { e.preventDefault(); void send(); }}>
          <Input.TextArea ref={input} value={draft} autoSize={{ minRows: 2, maxRows: 7 }} disabled={!ready}
            aria-label={tr("消息", "Message")} placeholder={tr("发送消息，继续追问…", "Message RWKV…")}
            onChange={e => { draftRef.current = e.target.value; setDraft(e.target.value); }} onBlur={saveDraft}
            onKeyDown={e => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing && e.keyCode !== 229) { e.preventDefault(); void send(); }
            }} />
          <div className="chat-composer-tools">
            <Button size="small" icon={<SettingOutlined />} onClick={() => setSettingsOpen(true)} disabled={!ready}
              aria-label={tr("检索设置", "Retrieval settings")}>{scopeLabel}</Button>
            <span className="chat-context-count">{tr(`历史 ${contextLength / 2} 轮`, `${contextLength / 2} context turns`)}</span>
            {busy ? <Button className="chat-send" type="primary" shape="circle" icon={<StopOutlined />}
              aria-label={tr("停止等待", "Stop waiting")} onClick={() => stop()} />
              : <Button className="chat-send" type="primary" shape="circle" htmlType="submit" icon={<ArrowUpOutlined />}
                disabled={!ready || !!issue} aria-label={tr("发送消息", "Send message")} />}
          </div>
        </form>
        {issueText && <p className="chat-input-warning" role="status">{issueText}</p>}
        {busy && busy.conversationId !== activeId && <p className="chat-input-warning" role="status">{tr("另一对话正在请求。请等待完成，或停止等待后再发送。", "Another chat has a pending request. Wait or stop waiting before sending.")}</p>}
        <p className="chat-composer-hint">{tr("Enter 发送 · Shift+Enter 换行 · 回答可能有误，请核对来源", "Enter to send · Shift+Enter for a new line · Answers may be incorrect; check sources")}</p>
      </div>
    </main>
    <Modal title={tr("删除此对话？", "Delete this chat?")} open={!!deletingId} onCancel={() => setDeletingId(undefined)}
      onOk={removeConversation} okText={tr("删除", "Delete")} cancelText={tr("取消", "Cancel")} okButtonProps={{ danger: true }}>
      {tr("只删除此浏览器内的会话；不会删除服务端运行记录。待处理请求将停止等待，不保证服务端取消。", "Only this browser's chat is deleted, not server traces. Waiting on its request will stop; server cancellation is not guaranteed.")}
    </Modal>
    <Modal title={tr("检索设置", "Retrieval settings")} open={settingsOpen} onCancel={() => setSettingsOpen(false)} footer={null}>
      <div className="chat-settings">
        <p>{tr("范围按会话固定。更换知识库或联网权限，请新建对话，避免混入其他范围的历史。", "Scope is fixed per chat. Start a new chat to change knowledge bases or web permissions without mixing histories.")}</p>
        <label htmlFor="chat-scope">{tr("资料渠道", "Source channels")}</label>
        <Select id="chat-scope" value={active?.scope.retrieval_mode} options={modes} disabled={!!active?.turns.length}
          onChange={value => changeScope({ retrieval_mode: value })} />
        <label htmlFor="chat-kb">{tr("知识库", "Knowledge base")}</label>
        <Select id="chat-kb" allowClear value={active?.scope.knowledge_base_id} disabled={!!active?.turns.length || active?.scope.retrieval_mode === "web"}
          placeholder={tr("全部知识库", "All knowledge bases")} options={knowledgeBases.map(k => ({ value: k.id, label: k.name }))}
          onChange={value => changeScope({ knowledge_base_id: value })} />
        {kbError && <Alert type="warning" title={tr("知识库列表加载失败", "Could not load knowledge bases")} description={kbError} />}
        <Alert type="info" title={tr("自动 / 网络模式可能向外部搜索服务发送模型生成的查询。请勿输入可外发任务不应包含的敏感内容。", "Auto / web modes may send model-generated queries to external search services. Do not include sensitive content in web-enabled tasks.")} />
        <p>{tr("会话与完整响应保存在此浏览器；请求仍按服务端策略留存。请勿在共享设备留下敏感资料。", "Chats and complete responses are stored in this browser; requests remain subject to server retention. Avoid leaving sensitive data on shared devices.")}</p>
      </div>
    </Modal>
  </div>;
}
