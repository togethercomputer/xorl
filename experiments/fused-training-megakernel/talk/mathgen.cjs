#!/usr/bin/env node
/* Pre-render the deck's math (LaTeX -> self-contained SVG, currentColor, sized
 * in ex so it scales with the surrounding font) and inline it into index.html
 * as the <script id="mkgap-mathsvg" type="application/json"> blob.
 *
 * Idempotent, same pattern as build.py. Needs mathjax-full@3 on the module
 * path: run from a directory whose node_modules has it, or set MATHJAX_NM.
 *   cd <dir with node_modules> && node /path/to/talk/mathgen.cjs
 */
"use strict";
const fs = require("fs");
const path = require("path");

const NM = process.env.MATHJAX_NM || path.resolve(process.cwd(), "node_modules");
const mj = p => require(path.join(NM, "mathjax-full/js", p));

const { mathjax } = mj("mathjax.js");
const { TeX } = mj("input/tex.js");
const { SVG } = mj("output/svg.js");
const { liteAdaptor } = mj("adaptors/liteAdaptor.js");
const { RegisterHTMLHandler } = mj("handlers/html.js");
const { AllPackages } = mj("input/tex/AllPackages.js");

// ---- the formula table: single source of truth for every typeset formula ----
const M = {
  // slide 1 · the linear-layer triple
  lin_f: "y = x\\,W^{\\top}",
  grad_def: "dz \\equiv \\partial \\ell / \\partial z",
  lin_dx: "dx = dy\\,W",
  lin_dw: "dW = dy^{\\top} x",

  // embedding
  emb_f: "x = P[\\mathrm{tok}]",
  emb_b: "dP[\\mathrm{tok}] \\mathrel{+}= dx",

  // rmsnorm (three sites share the backward-dx form)
  rms_r: "r = \\bigl(\\operatorname{mean}_H(x^2)+\\varepsilon\\bigr)^{-1/2}",
  rms_f: "\\hat{x} = \\frac{x}{\\sqrt{\\operatorname{mean}_H(x^2)+\\varepsilon}} \\odot w",
  rms_row: "\\hat{x} = x\\,\\bigl(\\operatorname{mean}_H(x^2)+\\varepsilon\\bigr)^{-1/2} \\odot w",
  rms_f2: "\\hat{x} = x \\odot r \\odot w_1",
  rms_bg: "g = dy \\odot w_1",
  rms_bdx: "dx = r \\odot \\bigl(g - \\hat{x}\\odot\\operatorname{mean}_H(g\\odot\\hat{x})\\bigr)",
  rms_bdw: "dw_1 = \\textstyle\\sum_S\\, dy \\odot \\hat{x}",
  rms2_f: "\\hat{x} = x \\odot r \\odot w_2",
  rms2_bdw: "dw_2 = \\textstyle\\sum_S\\, dy \\odot \\hat{x}",
  rmsf_f: "\\hat{x} = x \\odot r \\odot w_f",
  rmsf_bdw: "dw_f = \\textstyle\\sum_S\\, dy \\odot \\hat{x}",

  // qkv projection
  qkv_f: "[\\,q \\mid k \\mid v\\,] = \\hat{x}\\,W_{qkv}^{\\top}",
  qkv_bdx: "d\\hat{x} = d[\\,q \\mid k \\mid v\\,]\\,W_{qkv}",
  qkv_bdw: "dW_{qkv} = d[\\,q \\mid k \\mid v\\,]^{\\top}\\hat{x}",

  // qk-norm + rope
  rope_f: "q \\leftarrow R_{\\theta}\\,(\\operatorname{rms}(q)\\odot w_q)",
  rope_b: "dq_{\\text{pre}} = R_{\\theta}^{\\top}\\, dq",

  // attention
  attn_fS: "S = qK^{\\top}\\!/\\sqrt{D} + \\text{mask}",
  attn_fP: "P = \\operatorname{softmax}(S), \\qquad o = P\\,V",
  attn_f: "o = \\operatorname{softmax}\\!\\bigl(qK^{\\top}\\!/\\sqrt{D} + \\text{mask}\\bigr)\\,V",
  attn_drow: "D_{\\mathrm{row}} = \\operatorname{rowsum}(dO \\odot O)",
  attn_bdv: "dV = P^{\\top} dO, \\qquad dP = dO\\,V^{\\top}",
  attn_bds: "dS = P \\odot (dP - D_{\\mathrm{row}})",
  attn_bdqk: "dQ = dS\\,K/\\sqrt{D}, \\qquad dK = dS^{\\top} q/\\sqrt{D}",
  attn_b: "dV = P^{\\top}dO,\\;\\; dS = P\\odot(dO\\,V^{\\top}\\! - D_{\\mathrm{row}}),\\;\\; dQ = dS\\,K,\\;\\; dK = dS^{\\top} q",

  // o projection
  oproj_f: "x \\leftarrow x + o\\,W_o^{\\top}",
  oproj_bdx: "do = dy\\,W_o",
  oproj_bdw: "dW_o = dy^{\\top} o",

  // mlp
  mlp_f: "[\\,g \\mid u\\,] = \\hat{x}\\,W_{gu}^{\\top}, \\qquad h = \\operatorname{silu}(g)\\odot u",
  mlp_f2: "x \\leftarrow x + h\\,W_d^{\\top}",
  gu_f: "[\\,g \\mid u\\,] = \\hat{x}\\,W_{gu}^{\\top}",
  gu_bdx: "d\\hat{x} = d[\\,g \\mid u\\,]\\,W_{gu}",
  gu_bdw: "dW_{gu} = d[\\,g \\mid u\\,]^{\\top}\\hat{x}",
  swi_f: "h = \\operatorname{silu}(g)\\odot u, \\qquad \\operatorname{silu}(g)=g\\,\\sigma(g)",
  swi_row: "h = \\operatorname{silu}(g)\\odot u",
  swi_bdg: "dg = dh \\odot u \\odot \\sigma(g)\\,\\bigl(1+g\\,(1-\\sigma(g))\\bigr)",
  swi_bdu: "du = dh \\odot \\operatorname{silu}(g)",
  swiglu_b: "dg = dh\\odot u\\odot\\operatorname{silu}'(g),\\;\\; du = dh\\odot\\operatorname{silu}(g)",
  down_f: "x \\leftarrow x + h\\,W_d^{\\top}",
  down_bdx: "dh = dy\\,W_d",
  down_bdw: "dW_d = dy^{\\top} h",

  // lm head + loss
  head_f: "\\text{logits} = \\hat{x}\\,W_{lm}^{\\top}",
  head_bdx: "d\\hat{x} = d\\text{logits}\\;W_{lm}",
  head_bdw: "dW_{lm} = d\\text{logits}^{\\top}\\,\\hat{x}",
  ce_f: "\\ell = -\\tfrac{1}{S}\\textstyle\\sum_S \\log \\operatorname{softmax}(\\text{logits})[\\text{next}]",
  ce_row: "\\ell = -\\log p(\\text{next})",
  ce_b: "d\\text{logits} = \\bigl(\\operatorname{softmax}(\\text{logits}) - \\mathbf{1}_{\\text{next}}\\bigr)/S",
};

// ---- render ----
const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const doc = mathjax.document("", {
  InputJax: new TeX({ packages: AllPackages }),
  OutputJax: new SVG({ fontCache: "local" }),
});

const out = {};
for (const [k, texSrc] of Object.entries(M)) {
  const node = doc.convert(texSrc, { display: false, em: 16, ex: 8, containerWidth: 8000 });
  const svg = adaptor.innerHTML(node);
  if (svg.includes("data-mjx-error") || svg.includes("merror")) {
    throw new Error(`TeX error in '${k}': ${texSrc}`);
  }
  out[k] = svg;
}

// ---- inject ----
const htmlPath = path.join(__dirname, "index.html");
let html = fs.readFileSync(htmlPath, "utf8");
const pat = /(<script id="mkgap-mathsvg" type="application\/json">).*?(<\/script>)/s;
if (!pat.test(html)) throw new Error("mkgap-mathsvg blob not found in index.html");

// every key referenced in the page must exist in the table
const used = new Set(
  [...html.matchAll(/data-k="([a-z0-9_]+)"/g)].map(m => m[1])
    .concat([...html.matchAll(/\[\["?([a-z0-9_]+)"/g)].map(m => m[1]))
);
const missing = [...used].filter(k => !(k in out) && /_/.test(k));
if (missing.length) throw new Error("keys referenced but not in table: " + missing.join(", "));

const blob = JSON.stringify(out);
html = html.replace(pat, (_, a, b) => a + blob + b);
fs.writeFileSync(htmlPath, html);
console.log(`rendered ${Object.keys(out).length} formulas, inlined ${(blob.length / 1024).toFixed(0)} KB`);
