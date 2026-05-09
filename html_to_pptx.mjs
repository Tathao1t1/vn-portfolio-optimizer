/**
 * Convert poster.html → poster.pptx
 * 1. Screenshot HTML at 1920×1080 via puppeteer
 * 2. Embed the image as a full-bleed slide via pptxgenjs
 */
import puppeteer from "puppeteer";
import PptxGenJS from "pptxgenjs";
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import path from "path";
import { writeFileSync } from "fs";

const __dir  = path.dirname(fileURLToPath(import.meta.url));
const HTML   = path.join(__dir, "poster.html");
const SHOT   = path.join(__dir, "_poster_shot.png");
const OUT    = path.join(__dir, "poster.pptx");
const W = 1920, H = 1080;

// ── 1. Screenshot ──────────────────────────────────────────────────────────
console.log("📸  Screenshotting poster.html …");
const browser = await puppeteer.launch({
  headless: "new",
  args: ["--no-sandbox", "--disable-setuid-sandbox"],
});
const page = await browser.newPage();
await page.setViewport({ width: W, height: H, deviceScaleFactor: 2 }); // 2× for crisp text
await page.goto(`file://${HTML}`, { waitUntil: "networkidle0" });
await new Promise(r => setTimeout(r, 1500)); // let Google Fonts load
const imgBuf = await page.screenshot({ type: "png", fullPage: false });
await browser.close();
writeFileSync(SHOT, imgBuf);
console.log(`   screenshot: ${(imgBuf.length / 1024).toFixed(0)} KB`);

// ── 2. Build PPTX ──────────────────────────────────────────────────────────
console.log("📊  Building poster.pptx …");
const pptx = new PptxGenJS();
pptx.defineLayout({ name: "WIDE", width: 13.33, height: 7.5 });
pptx.layout = "WIDE";

const slide = pptx.addSlide();
const b64   = readFileSync(SHOT).toString("base64");
slide.addImage({
  data: `data:image/png;base64,${b64}`,
  x: 0, y: 0, w: "100%", h: "100%",
});

await pptx.writeFile({ fileName: OUT });
console.log(`✅  Saved → ${OUT}`);
