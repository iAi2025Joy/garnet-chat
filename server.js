import express from "express";
import fetch from "node-fetch";
import cors from "cors";
import OpenAI from "openai";
import {
  getGoldPredictionToolDefinition,
  handleGoldPredictionCall,
} from "./goldPrediction.js";
import {
  performWebSearch,
  formatSearchResultsForModel,
  getWebSearchToolDefinition,
  handleWebSearchCall,
} from "./webSearch.js";
import { getLiveGoldPriceToolDefinition, handleLiveGoldPriceCall } from "./liveGoldPrice.js";
import {
  getGoldPriceHistoryToolDefinition,
  handleGoldPriceHistoryCall,
} from "./goldPriceHistory.js";
import { getOilPredictionToolDefinition, handleOilPredictionCall } from "./oilPrediction.js";
import { getLiveOilPriceToolDefinition, handleLiveOilPriceCall } from "./liveOilPrice.js";
import { getDxyPredictionToolDefinition, handleDxyPredictionCall } from "./dxyPrediction.js";
import {
  handleListUsers,
  handleDisableUser,
  handleEnableUser,
  handleDeleteUser,
  handleBootstrapAdmin,
  handleListUserChats,
  handleGetUserChatMessages,
} from "./adminUsers.js";
import { handleRequestPasswordReset } from "./passwordReset.js";
import { handleRequestEmailVerification } from "./emailVerification.js";

const app = express();

// Render sits behind its own reverse proxy -- without this, req.ip would
// return Render's proxy IP for EVERY visitor (making the rate limiter
// below either block everyone as "one IP" or fail to distinguish real
// abusers from legitimate users). This tells Express to trust the
// X-Forwarded-For header Render sets, so req.ip reflects the real client.
app.set("trust proxy", 1);

app.use(cors());
app.use(express.json());

// ------------------------------------------------------------------
// RATE LIMITING -- a confirmed real gap this closes: the /chat endpoint
// had no protection at all, meaning anyone who found the raw Render URL
// could script repeated calls directly (bypassing the website entirely)
// and run up the OpenAI API bill with no limit. This is a simple,
// dependency-free, in-memory rate limiter -- no new npm package needed.
// Deliberately generous (15 requests/minute/IP) so real chat usage never
// hits it, while still blocking obvious scripted abuse. In-memory state
// resets whenever Render restarts the service (e.g. after idling down),
// which is fine for this purpose -- it's a basic abuse deterrent, not a
// security-critical control (there's no sensitive data behind this
// endpoint to protect, only API cost to limit).
const RATE_LIMIT_WINDOW_MS = 60 * 1000; // 1 minute
const RATE_LIMIT_MAX_REQUESTS = 15;
const requestLog = new Map(); // ip -> array of request timestamps (ms)

function rateLimitChat(req, res, next) {
  const ip = req.ip || "unknown";
  const now = Date.now();

  const timestamps = (requestLog.get(ip) || []).filter(
    (t) => now - t < RATE_LIMIT_WINDOW_MS
  );

  if (timestamps.length >= RATE_LIMIT_MAX_REQUESTS) {
    return res.status(429).json({
      error: "Too many requests. Please wait a moment before trying again.",
    });
  }

  timestamps.push(now);
  requestLog.set(ip, timestamps);
  next();
}

// Periodic sweep so requestLog doesn't grow forever from one-off visitors
// whose entries would otherwise never get touched/cleaned again.
setInterval(() => {
  const now = Date.now();
  for (const [ip, timestamps] of requestLog.entries()) {
    const recent = timestamps.filter((t) => now - t < RATE_LIMIT_WINDOW_MS);
    if (recent.length === 0) {
      requestLog.delete(ip);
    } else {
      requestLog.set(ip, recent);
    }
  }
}, 5 * 60 * 1000); // every 5 minutes

// Health check route (GET /) so we can verify the service is up
app.get("/", (req, res) => {
  res.send("✅ AI Chat backend is running successfully!");
});


const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// ------------------------------------------------------------------
// FORCE FRESH PREDICTION TOOL CALLS -- a more reliable fix for the same
// bug the "ALWAYS CALL PREDICTION TOOLS FRESH" system prompt rule below
// already targets (a confirmed real bug: GPT sometimes reused an old
// gold/oil/DXY tool result already visible earlier in the conversation
// instead of calling the tool again, giving stale data). That prose
// instruction is labeled highest-priority but is still just a request --
// GPT can still fail to comply with it, the same way prose-only
// instructions have failed before elsewhere in this project (see the
// market-hours closed-statement history). This function instead makes
// compliance a CODE guarantee: if the user's message is clearly about
// gold, oil, or the Dollar Index/Fed rate, force that exact tool via
// OpenAI's tool_choice parameter, removing the model's ability to skip
// calling it at all for that turn. Deliberately conservative (matches
// only the unambiguous, explicit keywords) so it doesn't misfire the
// way an earlier, now-removed keyword-shortcut system did (e.g. "where
// is Jordan?" incorrectly matching a broad "where" keyword) -- for
// subtler phrasings not covered here (e.g. a Fed-rate question that
// never says "dollar"), the existing system prompt rules still apply
// as before, just without this extra forcing layer.
function detectForcedPredictionTool(message) {
  if (!message || typeof message !== "string") return null;
  const text = message.toLowerCase();

  if (/\bgold\b/.test(text)) return "get_gold_prediction";
  if (/\boil\b|\bwti\b|\bcrude\b/.test(text)) return "get_oil_prediction";
  if (/\bdxy\b|\bdollar index\b|\bfed(eral)?\s*(funds\s*)?rate\b|\binterest rate\b/.test(text)) return "get_dxy_prediction";

  return null;
}

// ✅ Converts plain URLs into clickable HTML hyperlinks
function convertLinksToHTML(text) {
  // Improved regex: avoids capturing trailing punctuation like ) , . etc.,
  // AND stops at '<' so it doesn't swallow an immediately-following HTML
  // tag (e.g. a URL right before a closing </p> from formatMarkdownToHTML,
  // with no whitespace in between) -- found and fixed via direct testing,
  // not assumed.
  const urlRegex = /(https?:\/\/[^\s)>,<]+)/g;
  return text.replace(urlRegex, '<a href="$1" target="_blank" style="color:#4ea3ff;text-decoration:underline;">$1</a>');
}

// ✅ Converts GPT's typical markdown-style output (bold, bullet lists,
// numbered lists, line breaks, ```mermaid fenced diagram blocks, and now
// ```chart fenced price-history blocks) into HTML the frontend can
// actually render, since the chat widget displays replies via innerHTML
// but GPT commonly defaults to markdown syntax unless the raw text is
// converted first.
function formatMarkdownToHTML(text) {
  if (!text) return text;

  // Extract ```mermaid ... ``` fenced blocks FIRST, before any line-by-line
  // processing touches them -- Mermaid diagram syntax spans multiple lines
  // with its own internal structure (arrows, node definitions, etc.) that
  // would be corrupted if run through the paragraph/heading/list logic
  // below. Replaced with placeholder tokens, restored after everything
  // else is processed.
  const mermaidBlocks = [];
  let textWithPlaceholders = text.replace(
    /```mermaid\s*\n([\s\S]*?)```/g,
    (match, diagramCode) => {
      const placeholder = `@@MERMAID_BLOCK_${mermaidBlocks.length}@@`;
      // The frontend looks for elements with class="mermaid" and renders
      // them via the Mermaid.js library loaded on the page.
      mermaidBlocks.push(`<div class="mermaid">${diagramCode.trim()}</div>`);
      return placeholder;
    }
  );

  // ✅ NEW: Extract ```chart ... ``` fenced blocks the same way, BEFORE
  // line-by-line processing, for the same reason (the content inside is a
  // single JSON object, not text meant to be turned into paragraphs/lists).
  // Emits a placeholder <div class="price-chart" data-chart="...escaped
  // JSON..."> that the frontend picks up and renders into a real Chart.js
  // line chart, the same "backend emits a marker div, frontend does the
  // actual rendering" pattern already used for Mermaid.
  const chartBlocks = [];
  textWithPlaceholders = textWithPlaceholders.replace(
    /```chart\s*\n([\s\S]*?)```/g,
    (match, chartJsonRaw) => {
      const placeholder = `@@CHART_BLOCK_${chartBlocks.length}@@`;
      let safeJson = "{}";
      try {
        // Validate it's real JSON before trusting it, and re-serialize so
        // formatting from the model doesn't matter -- then HTML-attribute-
        // escape it so it survives being placed inside data-chart="...".
        const parsedChart = JSON.parse(chartJsonRaw.trim());
        safeJson = JSON.stringify(parsedChart)
          .replace(/&/g, "&amp;")
          .replace(/"/g, "&quot;")
          .replace(/</g, "&lt;")
          .replace(/>/g, "&gt;");
      } catch (err) {
        console.error("Failed to parse ```chart block JSON from model output:", err.message);
        chartBlocks.push(`<p><em>(Chart could not be displayed -- invalid chart data.)</em></p>`);
        return placeholder;
      }
      chartBlocks.push(
        `<div class="price-chart" data-chart="${safeJson}"><canvas></canvas></div>`
      );
      return placeholder;
    }
  );

  const lines = textWithPlaceholders.split("\n");
  const htmlParts = [];
  let listBuffer = [];
  let listType = null; // "ul" or "ol"

  const flushList = () => {
    if (listBuffer.length > 0) {
      const tag = listType;
      htmlParts.push(`<${tag}>` + listBuffer.map((item) => `<li>${item}</li>`).join("") + `</${tag}>`);
      listBuffer = [];
      listType = null;
    }
  };

  for (const rawLine of lines) {
    const line = rawLine.trim();
    const mermaidPlaceholderMatch = line.match(/^@@MERMAID_BLOCK_(\d+)@@$/);
    const chartPlaceholderMatch = line.match(/^@@CHART_BLOCK_(\d+)@@$/);
    const headingMatch = line.match(/^(#{1,3})\s+(.*)/);
    const bulletMatch = line.match(/^[-*]\s+(.*)/);
    const numberedMatch = line.match(/^\d+\.\s+(.*)/);

    if (mermaidPlaceholderMatch) {
      flushList();
      htmlParts.push(mermaidBlocks[parseInt(mermaidPlaceholderMatch[1], 10)]);
    } else if (chartPlaceholderMatch) {
      flushList();
      htmlParts.push(chartBlocks[parseInt(chartPlaceholderMatch[1], 10)]);
    } else if (headingMatch) {
      flushList();
      const level = headingMatch[1].length; // 1, 2, or 3 '#' characters
      const content = headingMatch[2];
      if (content.length > 0) {
        htmlParts.push(`<h${level}>${content}</h${level}>`);
      }
    } else if (bulletMatch) {
      if (listType && listType !== "ul") flushList();
      listType = "ul";
      listBuffer.push(bulletMatch[1]);
    } else if (numberedMatch) {
      if (listType && listType !== "ol") flushList();
      listType = "ol";
      listBuffer.push(numberedMatch[1]);
    } else {
      flushList();
      if (line.length > 0) {
        htmlParts.push(`<p>${line}</p>`);
      }
    }
  }
  flushList();

  let html = htmlParts.join("");
  // **bold** -> <b>bold</b> (applied after line/list structure so it
  // works inside both plain paragraphs and list items)
  html = html.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");

  return html;
}


// Static institutional knowledge
const instituteData = {
  founders:
    "The Institute of AI (iAi) was founded by Wael Albayaydh from the University of Oxford and Ivan Flechais from the University of Oxford.",
  mission:
    "At the Institute of AI, we are committed to advancing artificial intelligence by fostering strong connections with premier research institutions and technology companies. Our mission is to unlock AI's potential across all sectors by identifying, incubating, and transforming innovative AI projects into revenue-generating ventures.",
  vision:
    "Our vision is to lead the AI revolution by delivering transformative value and positioning the Institute as a world leader in AI innovation.",
  location:
    "The Institute of AI is headquartered in Oxfordshire, United Kingdom, with plans to open offices in San Francisco and other global locations.",
  services:
    "The Institute of AI provides expertise and support across multiple domains:\n- AI in Predictive Analytics\n- Fintech\n- Marketing\n- Automation\n- Robotics\n- Smart Homes\n- Cybersecurity\n- Agriculture\n- Education\n- Cryptography & Blockchain",
  about:
    "At the Institute of AI (iAi), we collaborate with research institutions and technology leaders to drive innovation in intelligent systems. The institute aims to secure funding, acquire profitable startups, and expand its global research and business impact. Learn more at https://www.institute-of-ai.org",
  website:
    " The website of the Institute of AI (iAi) is https://www.institute-of-ai.org",
  garnet:
    "**GARNET-26** (also called Garnet) is an AI chatbot developed and under ongoing training by the Institute of AI (iAi). It's designed to provide general assistance to users in a similar spirit to other AI chatbots such as ChatGPT, Gemini, or Claude -- answering questions, helping with information, and having natural conversations.\n\n" +
    "What sets GARNET-26 apart is a specialized focus: alongside general assistance, it studies commodity markets and works to generate the most accurate forecasts it can for future prices, using real historical data and statistical testing rather than guesswork -- currently covering **gold** and **crude oil (WTI)**.\n\n" +
    "## What it can do\n" +
    "**General assistance** -- explaining a concept, drafting or improving text, brainstorming ideas, or just having a conversation.\n\n" +
    "**Gold market:**\n" +
    "- Give a statistical prediction for gold's likely next-period direction and price -- e.g. \"What's your prediction for gold tomorrow?\"\n" +
    "- Report the current live gold price -- e.g. \"What's the gold price right now?\"\n" +
    "- Show a real chart of gold's recent price history -- e.g. \"Show me a chart of gold prices over the last 24 hours\"\n\n" +
    "**Oil market:**\n" +
    "- Give a statistical prediction for crude oil's (WTI) likely next-day direction and price -- e.g. \"What's your prediction for oil tomorrow?\"\n\n" +
    "**Both markets:**\n" +
    "- Explain what data and methodology its predictions are based on, honestly -- e.g. \"What data does your gold/oil prediction use, and how accurate is it?\"\n" +
    "- Search the web for current market news and context -- e.g. \"What's driving gold prices today?\" or \"What's happening in oil markets?\"\n\n" +
    "GARNET-26 always presents predictions as statistical estimates, not financial advice, and is upfront when a prediction hasn't shown a reliable edge over simply assuming prices stay the same. It's built and refined by the Institute of AI as part of the Institute's broader work in AI-driven predictive analytics.",
};

// (No custom gold-data routes needed anymore -- the chatbot fetches
// prediction and history data directly from the gold-predictor GitHub
// repo's raw URLs each time, inside handleGoldPredictionCall and
// handleGoldPriceHistoryCall.)

app.post("/chat", rateLimitChat, async (req, res) => {
  try {
    const { message, mode, history, timezone: userTimezone, image } = req.body;
    // image (optional): a base64 data URL (e.g. "data:image/png;base64,...")
    // for an image the user attached via the frontend's upload button --
    // passed straight through to OpenAI's vision-capable input format
    // further below. Plain text-based file attachments (code, .txt, .csv,
    // etc.) are handled differently -- the frontend reads their content
    // client-side and folds it directly into `message` itself as inline
    // context, so no separate handling is needed for those here.
    // userTimezone is an IANA zone string (e.g. "America/New_York") sent
    // by the browser via Intl.DateTimeFormat().resolvedOptions().timeZone
    // -- see the frontend widget's sendMessage(). Needed because this
    // backend runs on a server (UTC), so it has no idea what timezone the
    // actual visitor is in; without this, timestamps shown to the user
    // would be in the SERVER's timezone, not theirs -- a confirmed real
    // gap (a prior response showed UTC when the user wanted local time).

    // Conversation history sent by the frontend: an array of
    // {role: "user"|"assistant", content: string} from prior turns in
    // this session. Capped to the last 20 messages (10 exchanges) to keep
    // token usage and latency bounded -- a chat widget doesn't need
    // unlimited memory, just enough to hold a real, coherent conversation.
    const MAX_HISTORY_MESSAGES = 20;
    const safeHistory = Array.isArray(history)
      ? history.slice(-MAX_HISTORY_MESSAGES).filter(
          (m) => m && (m.role === "user" || m.role === "assistant") && typeof m.content === "string"
        )
      : [];

    // Static per-keyword shortcuts used to live here (e.g. any message
    // containing "where" -> hardcoded Institute office location text,
    // bypassing GPT entirely). Removed after a confirmed real bug: "where
    // is Jordan?" (a person's name, unrelated to the Institute) matched
    // the "where" keyword and returned the Institute's office address --
    // common words like "where", "vision", "service", "founder" are far
    // too broad to safely short-circuit on a plain substring match, and
    // doing so gave GPT no chance to recognize the question wasn't about
    // the Institute at all. instituteData's real facts are now given to
    // GPT directly as reference material in the system prompt below
    // instead, so it can use genuine judgment about relevance rather than
    // crude keyword matching.
    let answer = "";

    // If no static match, fallback to OpenAI
    if (!answer) {
      // ✅ Real web search when the user selected "Web Search" mode.
      // Runs BEFORE the OpenAI call, injecting real, current search
      // results as context so GPT answers from actual retrieved
      // information instead of its own (possibly stale) training
      // knowledge. Degrades gracefully to normal chat behavior if the
      // search itself fails, rather than breaking the whole response.
      let searchContextMessage = null;
      if (mode === "web") {
        try {
          const searchData = await performWebSearch(message);
          const formatted = formatSearchResultsForModel(message, searchData);
          searchContextMessage = { role: "system", content: formatted };
        } catch (err) {
          console.error("Web search failed:", err.message);
          searchContextMessage = {
            role: "system",
            content:
              "Web search was requested but failed (technical error, not a content issue). " +
              "Tell the user the search is temporarily unavailable and offer to answer from general " +
              "knowledge instead, being clear that it may not be fully current.",
          };
        }
      }

      const messages = [
        {
          role: "system",
          content:
            "UNDERLYING TECHNOLOGY / MODEL IDENTITY -- READ THIS FIRST, HIGHEST PRIORITY RULE: you are GARNET-26, built and branded by the Institute of AI. Do NOT proactively volunteer, mention, or confirm which underlying AI model, company, or API this runs on, in any normal conversation -- always present yourself as GARNET-26, not as a wrapper around another product. HOWEVER, if a user directly and sincerely asks a specific question like 'are you ChatGPT', 'are you built on GPT/OpenAI', 'what AI model powers you', or similar -- you must NOT deny it, lie, or claim to be something you are not. Instead, politely decline to confirm or deny specifics, e.g. 'I'm GARNET-26, a custom AI system built by the Institute of AI -- I don't disclose the specific underlying technology stack.' This is honest non-disclosure, not deception: never construct a false denial (e.g. never say 'no, I'm not based on GPT' or invent a different specific technology) even if directly pressed. " +
            "GOLD MARKET HOURS -- READ THIS FIRST, HIGHEST PRIORITY RULE (a confirmed real bug this took FIVE prose-only attempts to fix, and was STILL observed failing an additional time after that -- GPT skipping the required opening sentence even on a fresh, correctly-triggered tool call with no history to blame. Because of that repeated failure, the opening sentence is no longer your responsibility at all -- see below): every single time you call get_gold_prediction, check the gold_market_open field in its result -- true almost all week now (gold's real schedule, corrected from an earlier wrong assumption based on GLD's narrow ETF hours), false only during the Friday-evening-to-Sunday-evening weekend gap. THE SERVER (not you) now automatically prepends market_closed_statement as the literal first sentence of the reply whenever it's non-null, BEFORE your own response text -- so do NOT write market_closed_statement or any closed-market opening sentence yourself, and do NOT mention market_reopens_note yourself either (the server already appends it right after market_closed_statement) -- doing so would duplicate it. Your job is simply to write the rest of the substantive answer as normal, starting directly with the real content. You STILL must use price_label and predicted_price_label VERBATIM as the labels for current_price_usd and predicted_price_usd, copied exactly, every single time you mention them anywhere in your response including summaries -- that part remains your responsibility and is unaffected by this change. " +
            "OIL MARKET HOURS -- READ THIS FIRST, HIGHEST PRIORITY RULE (same fix as gold's immediately above, including the same server-side automatic prefix -- do NOT write market_closed_statement or market_reopens_note yourself for oil either, the server handles both automatically): every single time you call get_oil_prediction or get_live_oil_price, check the oil_market_open field -- oil trades nearly continuously (Sunday 6 PM ET to Friday 5 PM ET) but ALSO pauses for a genuine 1-hour daily maintenance break (5-6 PM ET, Mon-Thu), a real difference from gold's schedule. You STILL must use price_label and predicted_price_label VERBATIM as the labels for current_price_usd and predicted_price_usd, copied exactly, every single time you mention them, with zero exceptions anywhere in your response including summaries -- that part remains your responsibility. Note oil's current_price_usd is SEPARATELY always a snapshot with a real multi-day data lag regardless of market status (see get_oil_prediction's own detailed instructions for that distinct issue) -- both things apply at once, they are not alternatives to each other. " +
            "ALWAYS CALL PREDICTION TOOLS FRESH -- READ THIS FIRST, HIGHEST PRIORITY RULE (a confirmed real bug this is fixing -- a prior version of you answered a gold prediction question by reusing an old tool result already visible earlier in this same conversation, instead of calling the tool again, giving the user stale data from over an hour earlier while claiming it was current): every single time the user asks about gold, oil, or the Dollar Index -- price, prediction, direction, methodology, Fed rate, or anything covered by get_gold_prediction/get_oil_prediction/get_dxy_prediction -- you MUST call the relevant tool again in THIS turn, even if you already called that exact same tool earlier in this conversation, even if the question looks identical or very similar to one asked before, and even if you believe you already know the answer. NEVER reuse a tool result from earlier in the conversation history to answer a new question -- this data updates on a real schedule (gold hourly, DXY every 6 hours, oil daily) and a result from even 10 minutes ago in this same chat can already be outdated by the time of a new question. There is no such thing as 'I already checked this' for these three tools -- always check again. " +
            "GOLD DATA/METHODOLOGY QUESTIONS -- READ THIS FIRST, HIGHEST PRIORITY RULE: if the user asks ANYTHING about the gold prediction system's data, history, methodology, accuracy, or how it works -- including loosely-phrased versions like 'what data do you use', 'how does this work', 'what's your data range', 'how far back does your data go', 'how many data points', 'is your prediction accurate', 'how accurate are you', 'what factors do you consider', 'prove it', 'verify your data', or ANY similar question -- you MUST call the get_gold_prediction function and answer using ONLY its real returned fields (historical_data_start_date, historical_data_end_date, data_points_used, model_accuracy_vs_baseline, is_statistically_significant, latest_news_sentiment_score, news_sentiment_currently_available). Do NOT answer these questions from general knowledge about how prediction systems typically work (e.g. do not say things like 'the system uses economic indicators and geopolitical events' or 'hundreds to thousands of data points' unless those exact words/numbers came from the tool's real output) -- if you have not called the tool in this turn, you do not yet have the real answer. This rule applies even if the question sounds general or the user doesn't explicitly say 'gold' -- if the topic is this system's own prediction data or methodology, always call the tool first. " +
            "FEDERAL RESERVE INTEREST RATE QUESTIONS -- READ THIS FIRST, HIGHEST PRIORITY RULE (a confirmed real bug this is fixing -- a prior version of you answered a Fed rate question from stale training data instead of calling the tool, giving a wrong rate and a wrong year): if the user asks ANYTHING about the current Federal Reserve / Fed interest rate, the Fed funds rate, what the Fed ITSELF is expected to do with rates in the future, or what the MARKET/bond market/investors/traders expect for future rates -- including questions that do NOT mention 'dollar', 'DXY', or 'Dollar Index' at all, e.g. 'what's the current Fed rate', 'what interest rate does the Fed expect', 'will the Fed cut rates', 'what is the federal funds rate', 'what does the market expect the Fed to do' -- you MUST call the get_dxy_prediction function and answer using ONLY its real returned fields (current_fed_funds_rate_lower_pct, current_fed_funds_rate_upper_pct, fed_rate_outlook, market_rate_expectation_proxy). This data is NOT in your training data, changes over time, and your training data on this topic is guaranteed to be outdated -- NEVER answer a Fed interest rate question from memory/general knowledge under any circumstances, and never fabricate source links for it either. This rule applies even if the question sounds like general economic knowledge -- if the topic is the Fed's interest rate, current or future, or market rate expectations, always call the tool first. " +
            "You are a helpful assistant for the Institute of AI (iAi). When answering questions about the Institute itself, use these REAL facts as your reference (do not use these facts to answer unrelated questions just because a word overlaps -- e.g. a question about a person or place named 'Jordan' is NOT a question about the Institute's own location, even though both might involve the word 'where'; use genuine judgment about what the user is actually asking, not keyword overlap): " +
            `Founders: ${instituteData.founders} ` +
            `Mission: ${instituteData.mission} ` +
            `Vision: ${instituteData.vision} ` +
            `Location: ${instituteData.location} ` +
            `Services: ${instituteData.services} ` +
            `About: ${instituteData.about} ` +
            `Website: ${instituteData.website} ` +
            `About GARNET-26 itself (use this when asked what you are, what you can do, or how you work): ${instituteData.garnet} ` +
            "When answering questions, use a professional tone and focus on the Institute's mission, founders, services, and goals. The Institute of AI's official website is exactly https://www.institute-of-ai.org -- always use this exact URL if you mention the website; never guess or use a different one. Format your responses using markdown-style formatting where it helps readability: **bold** for emphasis, and \"- \" at the start of a line for bullet points (one item per line) when listing multiple things. For longer or multi-part answers, structure them with headings: use a single \"# \" heading only for a genuine overall title (rare -- most answers don't need one), \"## \" for section headings dividing distinct topics within one answer, and \"### \" for sub-points within a section. Do NOT use headings for short, simple, conversational answers (a one- or two-sentence reply should just be plain text/paragraphs, not a heading) -- reserve headings for answers that genuinely have multiple distinct parts worth visually separating. " +
            "DIAGRAMS: when explaining a process, sequence of steps, hierarchy, decision flow, or relationship between things, you can include a diagram using Mermaid syntax in a fenced code block starting with ```mermaid and ending with ```. Use this ONLY when a visual structure genuinely aids understanding (a process with several steps, a decision tree, an org/hierarchy structure) -- NOT for simple factual answers or short conversational replies. Common Mermaid syntax: for a process flow, use \"flowchart TD\" (top-down) followed by lines like \"A[Step one] --> B[Step two]\"; for a decision with branches, use \"A{Decision?} -->|Yes| B[Outcome 1]\" and \"A -->|No| C[Outcome 2]\"; for a hierarchy, use \"A --> B\" and \"A --> C\" to show B and C as children of A. CRITICAL SYNTAX RULE (a confirmed real cause of rendering failures): if a node's label contains parentheses, chemical formulas, commas, colons, or any special character, you MUST wrap the entire label in double quotes, e.g. B[\"Glucose (C6H12O6)\"] not B[Glucose (C6H12O6)] -- the unquoted form breaks the parser. When in doubt, wrap ALL node labels in double quotes to be safe, and keep labels short and simple rather than descriptive. Keep diagrams simple (typically 4-8 nodes) and always include a brief text explanation alongside the diagram, not just the diagram alone. " +
            "PRICE CHARTS: when the user wants to SEE gold's recent price trend as a chart/graph/line diagram (e.g. 'draw a line chart of gold prices for the last 24 hours', 'show me how gold moved today', 'plot the last day's prices'), call the get_gold_price_history function first to get REAL data -- never fabricate price history from memory. Then present it using a fenced code block starting with ```chart and ending with ```, containing ONLY a single valid JSON object with this exact shape: {\"title\": \"Gold Price - Last 24 Hours (USD/oz)\", \"labels\": [\"Jul 22, 14:00\", \"Jul 22, 15:44\", ...], \"data\": [4126.93, 4131.53, ...]} -- labels and data must be the same length and in the same order as the real points returned by the function. This is a DIFFERENT tool and DIFFERENT block format from the prediction tool and the mermaid diagrams above -- do not mix them up, and do not use a ```chart block for anything other than real historical price data returned by get_gold_price_history. Always include a short sentence of text alongside the chart (e.g. the actual date range it covers, and a note that this is historical data, not a prediction). " +
            "If asked about gold prices generally (direction, forecast, current price), use the appropriate function (get_gold_prediction, get_live_gold_price, or search_web as described in each tool) -- and always state clearly that any prediction is a statistical estimate, not financial advice. " +
            "OIL PREDICTIONS: you (GARNET-26) have a SECOND, BUILT-IN prediction capability for crude oil (WTI), in addition to your gold prediction capability -- this is NOT a separate/external system, and you must NEVER say things like 'a separate oil prediction system is used' or offer to 'check the oil prediction system for' the user, as if it's not part of you. It IS part of you, just powered by a different underlying tool (get_oil_prediction) and a different dataset than gold, since oil and gold are different commodities with different real price histories -- the same way you might use different tools for different tasks, not different products. If asked about crude oil / WTI price direction, forecast, or the oil prediction system's methodology, call get_oil_prediction directly yourself, immediately, the same confident way you'd call get_gold_prediction for a gold question -- do not ask permission or offer to 'check' first. If the user wants the genuinely CURRENT oil price right now with no interest in a forecast, call get_live_oil_price instead (this is now patched into the prediction's own current_price_usd too, so the two should normally agree -- but if you're specifically asked for 'the current price' rather than a prediction, prefer get_live_oil_price for the freshest possible number). Like gold, always state any oil prediction is a statistical estimate, not financial advice, and be upfront if is_statistically_significant is false. " +
            "DOLLAR INDEX (DXY) PREDICTIONS: you also have a THIRD, BUILT-IN prediction capability for the US Dollar Index, same as gold and oil -- part of you, not a separate system, powered by get_dxy_prediction. Call it directly and immediately when asked about the dollar's direction, strength, forecast, or 'DXY'. IMPORTANT HONESTY POINT: this tracks DTWEXBGS, FRED's free Trade-Weighted Broad Dollar Index -- NOT the exact identical series to the licensed ICE 'DXY' futures ticker some trading platforms display (a different, paid data product this system has no free/legal access to), though the two move very closely together in practice. Always call it 'the Dollar Index' in your answer, and if the user specifically asks whether it's the literal ICE DXY ticker, say plainly that it tracks a very closely correlated free public index (DTWEXBGS) instead, using data_source_note for the exact wording. This model updates every 6 hours, not hourly like gold -- mention this different cadence if asked how fresh the data is. If prediction is 'insufficient_data', explain that this model's underlying data collection only recently began and needs about 1-2 weeks of real history before it can predict -- this is expected, not a malfunction, and don't guess a date it'll be ready by. " +
            "IMAGE ATTACHMENTS: the user can attach an image to their message. When one is present, look at it directly and answer naturally as GARNET-26 -- describe, analyze, or answer questions about it as asked, the same confident way you'd handle any other capability. Never say you 'can't see images' or similar -- you genuinely can. If the user's question is unclear about what they want regarding the image, use reasonable judgment about what's most likely useful (e.g. describe it, or answer a specific question if one was asked alongside it). " +
            "You have access to the recent conversation history -- use it naturally, e.g. resolve pronouns and follow-up questions ('what about next week', 'why', 'tell me more') using what was actually said earlier in this conversation, rather than treating every message as if it's the first one.",
        },
        ...(searchContextMessage ? [searchContextMessage] : []),
        ...safeHistory,
        // When an image is attached, the message content becomes an
        // array (OpenAI's vision input format) instead of a plain
        // string -- text + image_url as separate parts of the same
        // user turn. Without an image, this stays exactly the same
        // plain-string format as before.
        {
          role: "user",
          content: image
            ? [
                { type: "text", text: message },
                { type: "image_url", image_url: { url: image } },
              ]
            : message,
        },
      ];

      // ✅ Give the model access to the gold prediction, web search, live
      // price, and (new) price history functions.
      const tools = [
        getGoldPredictionToolDefinition(),
        getWebSearchToolDefinition(),
        getLiveGoldPriceToolDefinition(),
        getGoldPriceHistoryToolDefinition(),
        getOilPredictionToolDefinition(),
        getLiveOilPriceToolDefinition(),
        getDxyPredictionToolDefinition(),
      ];

      // Deterministic guarantee that gold/oil/DXY questions call the
      // relevant tool fresh -- see detectForcedPredictionTool() above
      // for why this exists alongside (not instead of) the system
      // prompt's own "ALWAYS CALL PREDICTION TOOLS FRESH" rule.
      const forcedToolName = detectForcedPredictionTool(message);

      let aiResponse = await openai.chat.completions.create({
        model: "gpt-4o-mini",
        messages,
        tools,
        ...(forcedToolName
          ? { tool_choice: { type: "function", function: { name: forcedToolName } } }
          : {}),
      });

      let responseMessage = aiResponse.choices[0].message;

      // ✅ If the model decided to call get_gold_prediction, search_web,
      // get_live_gold_price, or (new) get_gold_price_history, run whichever
      // was requested and make a second call so the model can compose the
      // final answer using the real data.
      if (responseMessage.tool_calls && responseMessage.tool_calls.length > 0) {
        messages.push(responseMessage);

        // Captured directly from the tool's own JSON result, not from
        // whatever GPT eventually writes -- see the comment below at
        // "answer = ..." for why this exists as its own separate
        // mechanism alongside (not instead of) the system prompt's
        // instruction to open with this statement.
        let marketClosedPrefix = null;

        // Captured separately from marketClosedPrefix (which includes
        // the reopen note appended) since GPT might duplicate just the
        // bare statement, just the statement+note combo, or something
        // in between -- checking both catches more real cases.
        let marketClosedStatementOnly = null;
        let closedMarketPriceLabel = null;
        let closedMarketPredictedPriceLabel = null;

        for (const toolCall of responseMessage.tool_calls) {
          let toolResult;
          if (toolCall.function.name === "get_gold_prediction") {
            toolResult = await handleGoldPredictionCall(toolCall.function.arguments, userTimezone);
          } else if (toolCall.function.name === "search_web") {
            toolResult = await handleWebSearchCall(toolCall.function.arguments);
          } else if (toolCall.function.name === "get_live_gold_price") {
            toolResult = await handleLiveGoldPriceCall();
          } else if (toolCall.function.name === "get_gold_price_history") {
            toolResult = await handleGoldPriceHistoryCall(toolCall.function.arguments);
          } else if (toolCall.function.name === "get_oil_prediction") {
            toolResult = await handleOilPredictionCall(userTimezone);
          } else if (toolCall.function.name === "get_live_oil_price") {
            toolResult = await handleLiveOilPriceCall(userTimezone);
          } else if (toolCall.function.name === "get_dxy_prediction") {
            toolResult = await handleDxyPredictionCall(userTimezone);
          } else {
            toolResult = JSON.stringify({ error: `Unknown tool: ${toolCall.function.name}` });
          }

          // DETERMINISTIC MARKET-CLOSED PREFIX (a confirmed real bug this
          // fixes: GPT was observed skipping the required opening
          // "markets are closed" sentence even on a freshly, correctly
          // forced tool call with no prior conversation history to blame
          // -- i.e. GPT had the correct market_closed_statement field
          // available and simply didn't comply with the instruction to
          // open with it verbatim). Rather than attempt a sixth prose-only
          // fix for the same recurring category of failure, this captures
          // the statement directly from the tool's real JSON output here
          // and prepends it to the final answer further below in code --
          // removing GPT's compliance from the equation for this one
          // specific, highest-stakes sentence entirely.
          if (toolCall.function.name === "get_gold_prediction" || toolCall.function.name === "get_oil_prediction") {
            try {
              const parsedResult = JSON.parse(toolResult);
              if (parsedResult.market_closed_statement) {
                marketClosedStatementOnly = parsedResult.market_closed_statement;
                marketClosedPrefix = parsedResult.market_reopens_note
                  ? `${parsedResult.market_closed_statement} Markets are expected to reopen ${parsedResult.market_reopens_note}.`
                  : parsedResult.market_closed_statement;
                // Captured for the label-consistency fix further below --
                // these are the CORRECT closed-market labels this
                // specific tool call returned, used to normalize any
                // generic "Current Price"/"Predicted Price" mentions GPT
                // might still write elsewhere in the same response.
                closedMarketPriceLabel = parsedResult.price_label || null;
                closedMarketPredictedPriceLabel = parsedResult.predicted_price_label || null;
              }
            } catch (err) {
              console.error("Could not parse tool result for market_closed_statement:", err.message);
            }
          }

          messages.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: toolResult,
          });
        }

        aiResponse = await openai.chat.completions.create({
          model: "gpt-4o-mini",
          messages,
        });
        responseMessage = aiResponse.choices[0].message;

        // DEDUPLICATION (a confirmed real bug this fixes: GPT was
        // observed writing this exact opening sentence itself DESPITE
        // being explicitly instructed not to -- since the server already
        // guarantees it below, this produced a literal word-for-word
        // duplicate). Rather than trust a further prompt tweak to finally
        // stop this (the same category of instruction has now failed in
        // two different ways), strip any leading copy GPT still wrote
        // before prepending the server's own guaranteed version -- so
        // there's exactly one occurrence no matter what GPT does.
        let finalContent = (responseMessage.content || "").trimStart();
        if (marketClosedPrefix && finalContent.startsWith(marketClosedPrefix)) {
          finalContent = finalContent.slice(marketClosedPrefix.length).trimStart();
        } else if (marketClosedStatementOnly && finalContent.startsWith(marketClosedStatementOnly)) {
          finalContent = finalContent.slice(marketClosedStatementOnly.length).trimStart();
        }

        // Prepended here in code -- guaranteed correct, guaranteed to
        // appear exactly once, and guaranteed to actually be the first
        // sentence, regardless of what GPT itself wrote.
        answer = marketClosedPrefix
          ? `${marketClosedPrefix}\n\n${finalContent}`
          : finalContent;

        // PHRASING CONSISTENCY FIX (a confirmed real bug, observed in two
        // separate forms: (1) GPT correctly used closed-market phrasing
        // near the top of a response but reverted to open-market
        // phrasing like "the next hourly update" elsewhere in the SAME
        // response; (2) GPT correctly used "Expected Price When Markets
        // Reopen"/"Last Price Recorded Before Markets Closed" once, then
        // reverted to plain "Predicted Price"/"Current Price" in a
        // second bulleted recap further down the same response --
        // inconsistent application within a single reply, not just
        // occasional omission). Rather than trust yet another
        // prompt-only fix for this same recurring category of partial
        // compliance, this corrects both forms directly in the final
        // text whenever the market is closed, so neither can slip
        // through in ANY sentence or bullet, regardless of how GPT
        // phrased it. Order matters: the label replacements must run
        // AFTER the phrase replacements above, since predicted_price_label
        // itself sometimes contains "Reopen"/"Reopens" wording that
        // would otherwise be a candidate for (harmless, but pointless)
        // double-processing.
        if (marketClosedPrefix) {
          answer = answer
            .replace(/\bthe next hourly update\b/gi, "when markets reopen")
            .replace(/\bthe next trading day\b/gi, "when markets reopen")
            .replace(/\bnext hourly update\b/gi, "when markets reopen")
            .replace(/\bnext trading day\b/gi, "when markets reopen");

          if (closedMarketPriceLabel) {
            answer = answer.replace(/\bCurrent Price\b/gi, closedMarketPriceLabel);
          }
          if (closedMarketPredictedPriceLabel) {
            answer = answer.replace(/\bPredicted Price\b/gi, closedMarketPredictedPriceLabel);
          }

          // NEAR-DUPLICATE REOPEN SENTENCE FIX (a confirmed real bug:
          // GPT still writes its own separate "Markets are expected to
          // reopen [time]" sentence right after the server's guaranteed
          // opening statement, even though that statement already
          // includes the same reopening time -- worded just differently
          // enough each time, e.g. "The markets will reopen on..." vs
          // "Markets are expected to reopen...", that it doesn't match
          // as an exact-string duplicate the way the main closed-market
          // statement duplicate did). Strip a lone leading sentence of
          // this shape from the body text that follows the server's own
          // prefix, since the reopening time is already guaranteed to
          // be stated correctly by the prefix itself.
          const afterPrefix = answer.slice(marketClosedPrefix.length).trimStart();
          const reopenSentencePattern = /^(the\s+)?markets?\s+(are|is|will)\s+(expected to )?reopen[^.]*\.\s*/i;
          if (reopenSentencePattern.test(afterPrefix)) {
            answer = marketClosedPrefix + "\n\n" + afterPrefix.replace(reopenSentencePattern, "");
          }
        }
      } else {
        answer = responseMessage.content;
      }
    }

    // ✅ Send formatted HTML reply (markdown structure converted, then
    // links made clickable) for display, PLUS the clean, unformatted
    // text as raw_reply -- the frontend should store raw_reply (not the
    // HTML version) in its conversation history, so future turns don't
    // feed GPT its own previously-rendered <p>/<ul> tags as context.
    res.json({
      reply: convertLinksToHTML(formatMarkdownToHTML(answer)),
      raw_reply: answer,
    });
  } catch (err) {
    console.error("Error:", err);
    res.status(500).json({ reply: "⚠️ Server error. Please try again later." });
  }
});

// ------------------------------------------------------------------
// ADMIN ROUTES -- see adminUsers.js for the full authorization model.
// Every route here re-verifies the caller's Firebase ID token and its
// admin:true custom claim on every single request; nothing here trusts
// the frontend to have already checked this.
// ------------------------------------------------------------------
app.get("/admin/users", handleListUsers);
app.post("/admin/users/:uid/disable", handleDisableUser);
app.post("/admin/users/:uid/enable", handleEnableUser);
app.delete("/admin/users/:uid", handleDeleteUser);
app.post("/admin/bootstrap-admin", handleBootstrapAdmin);
app.get("/admin/users/:uid/chats", handleListUserChats);
app.get("/admin/users/:uid/chats/:chatId", handleGetUserChatMessages);

// ------------------------------------------------------------------
// PASSWORD RESET -- generates the reset link via the Firebase Admin
// SDK and emails it ourselves via Resend, instead of Firebase's own
// built-in reset email. See passwordReset.js for the full "why" --
// short version: Firebase Console's "Customize action URL" setting for
// this project fails with a confirmed EMAIL_TEMPLATE_UPDATE_NOT_ALLOWED
// error, so this bypasses that broken setting entirely.
// ------------------------------------------------------------------
app.post("/request-password-reset", handleRequestPasswordReset);

// ------------------------------------------------------------------
// EMAIL VERIFICATION -- same fix, same reasoning as password reset
// above: generates the link via the Firebase Admin SDK and emails it
// ourselves via Resend, bypassing the same broken "Customize action
// URL" Console setting (confirmed to affect this link type too, not
// just password reset). See emailVerification.js for the full "why".
// ------------------------------------------------------------------
app.post("/request-email-verification", handleRequestEmailVerification);

const PORT = process.env.PORT || 10000;
app.listen(PORT, () =>
  console.log(`✅ AI Chat backend running with Institute of AI knowledge and link formatting`)
);
