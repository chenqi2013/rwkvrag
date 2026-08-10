import { createContext, useContext, useEffect, useMemo, useState } from "react";

export type Language = "zh" | "en";

interface LanguageContextValue {
  language: Language;
  setLanguage: (language: Language) => void;
  tr: (zh: string, en: string) => string;
}

const LanguageContext = createContext<LanguageContextValue | null>(null);

function initialLanguage(): Language {
  const saved = window.localStorage.getItem("rwkvrag-language");
  if (saved === "zh" || saved === "en") return saved;
  return window.navigator.language.toLowerCase().startsWith("zh") ? "zh" : "en";
}

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [language, setLanguage] = useState<Language>(initialLanguage);

  useEffect(() => {
    window.localStorage.setItem("rwkvrag-language", language);
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
    document.title = language === "zh" ? "RWKVRAG 知识库管理" : "RWKVRAG Knowledge Console";
  }, [language]);

  const value = useMemo<LanguageContextValue>(() => ({
    language,
    setLanguage,
    tr: (zh, en) => language === "zh" ? zh : en,
  }), [language]);

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
}

export function useLanguage() {
  const value = useContext(LanguageContext);
  if (!value) throw new Error("useLanguage must be used inside LanguageProvider");
  return value;
}
