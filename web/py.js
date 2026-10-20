// Python's string formatting, for the text the model reads. The tools speak to the model in the same words and
// numbers as the Python version, so a run in the browser and a run on the command line see the same things.

export function format(template, vars) {
  return template.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? String(vars[k]) : m));
}

const strip = (s) => (s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s);

// {x:g} and {x:.2g}
export function g(x, p = 6) {
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x > 0 ? "inf" : "-inf";
  if (x === 0) return Object.is(x, -0) ? "-0" : "0";
  const [mant, e] = x.toExponential(p - 1).split("e");
  const exp = Number(e);
  if (exp < -4 || exp >= p) return `${strip(mant)}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
  return strip(x.toFixed(Math.max(0, p - 1 - exp)));
}

// {x:.Nf}. Python rounds an exact half to even; toFixed rounds it away from zero.
export function f(x, d) {
  if (!Number.isFinite(x)) return g(x);
  const exact = Math.abs(x).toFixed(Math.min(100, d + 30));
  const dot = exact.indexOf(".");
  if (!/^50*$/.test(exact.slice(dot + 1 + d))) return x.toFixed(d);
  const kept = exact.slice(0, dot + 1 + d).replace(/\.$/, "");
  const last = Number(kept[kept.length - 1]);
  const down = last % 2 === 0;
  const mag = down ? kept : (Math.abs(x) + 0.5 * 10 ** -d).toFixed(d);
  return (x < 0 ? "-" : "") + mag;
}
export const signed = (x, d) => (x >= 0 ? "+" : "") + f(x, d);
export const pct = (x, d = 0) => f(x * 100, d) + "%";
// How Python prints a float on its own: 8.0 stays 8.0.
export const pyFloat = (x) => (Number.isInteger(x) ? x.toFixed(1) : String(x));
export const capitalize = (s) => (s ? s[0].toUpperCase() + s.slice(1).toLowerCase() : s);

const HIDDEN = /[\p{Cc}\p{Cf}\p{Cs}\p{Co}\p{Cn}\p{Zl}\p{Zp}\p{Zs}]/u;

// repr() of a str.
export function repr(s) {
  const q = s.includes("'") && !s.includes('"') ? '"' : "'";
  let out = q;
  for (const ch of s) {
    const c = ch.codePointAt(0);
    if (ch === "\\") out += "\\\\";
    else if (ch === q) out += "\\" + q;
    else if (ch === "\n") out += "\\n";
    else if (ch === "\r") out += "\\r";
    else if (ch === "\t") out += "\\t";
    else if (ch !== " " && HIDDEN.test(ch)) {
      out += c <= 0xff ? "\\x" + c.toString(16).padStart(2, "0") : c <= 0xffff ? "\\u" + c.toString(16).padStart(4, "0") : "\\U" + c.toString(16).padStart(8, "0");
    } else out += ch;
  }
  return out + q;
}

export function stamp(d = new Date()) {
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

// True when text compresses like a broken record, the usual sign of a damaged model. zlib at its default level, as in Python.
export async function babbling(text) {
  const raw = new TextEncoder().encode(text);
  if (raw.length <= 400) return false;
  const stream = new Blob([raw]).stream().pipeThrough(new CompressionStream("deflate"));
  const packed = await new Response(stream).arrayBuffer();
  return packed.byteLength / raw.length < 0.12;
}
