import { isConversation, storageRevisionMatches, type Conversation } from "./chatState.ts";

const DATABASE = "rwkvrag-chat-v1";
const STORE = "conversations";

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    let blocked = false;
    const request = indexedDB.open(DATABASE, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE, { keyPath: "id" });
    request.onerror = () => reject(request.error);
    request.onblocked = () => { blocked = true; reject(new Error("Browser storage is blocked by another tab.")); };
    request.onsuccess = () => {
      const database = request.result;
      if (blocked) { database.close(); return; }
      database.onversionchange = () => database.close();
      resolve(database);
    };
  });
}

export async function loadConversations(): Promise<{ conversations: Conversation[]; unreadable: number }> {
  const database = await openDatabase();
  try {
    return await new Promise((resolve, reject) => {
      const transaction = database.transaction(STORE, "readonly");
      const request = transaction.objectStore(STORE).getAll();
      transaction.oncomplete = () => {
        const rows: unknown[] = request.result;
        const conversations = rows.filter(isConversation);
        resolve({ conversations, unreadable: rows.length - conversations.length });
      };
      transaction.onabort = () => reject(transaction.error ?? request.error);
      transaction.onerror = () => reject(transaction.error ?? request.error);
    });
  } finally { database.close(); }
}

export async function persistConversation(conversation: Conversation, deleting = false): Promise<void> {
  const database = await openDatabase();
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(STORE, "readwrite");
      const store = transaction.objectStore(STORE);
      const request = store.get(conversation.id);
      let conflict = false;
      request.onsuccess = () => {
        if (!storageRevisionMatches(request.result, conversation, deleting)) {
          conflict = true;
          transaction.abort();
          return;
        }
        if (deleting) store.delete(conversation.id);
        else store.put(conversation);
      };
      transaction.oncomplete = () => resolve();
      transaction.onabort = transaction.onerror = () => reject(conflict
        ? new Error("Conversation changed in another tab; stored data was not overwritten.")
        : transaction.error ?? request.error ?? new Error("Browser storage write failed."));
    });
  } finally { database.close(); }
}
