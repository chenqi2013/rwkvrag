export function formatBytes(bytes: number): string {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function locale(): string {
  return document.documentElement.lang === "en" ? "en-US" : "zh-CN";
}

export function formatNumber(value: number): string {
  return new Intl.NumberFormat(locale()).format(value);
}

export function formatDate(value?: string): string {
  if (!value) return "—";
  return new Intl.DateTimeFormat(locale(), {
    dateStyle: "short",
    timeStyle: "medium",
  }).format(new Date(value));
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : locale() === "en-US" ? "Operation failed" : "操作失败";
}
