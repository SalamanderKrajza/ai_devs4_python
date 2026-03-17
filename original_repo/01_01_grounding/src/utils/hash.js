import { createHash } from "node:crypto";

const stableStringify = (value) => {
  if (Array.isArray(value)) {
    return `[${value.map(stableStringify).join(",")}]`;
  }

  if (value && typeof value === "object") {
    const keys = Object.keys(value).sort();
    const entries = keys.map(
      (key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`
    );
    return `{${entries.join(",")}}`;
  }

  return JSON.stringify(value);
};

export const hashText = (text) =>
  createHash("sha256").update(text).digest("hex");

export const hashObject = (obj) => hashText(stableStringify(obj));
