#!/usr/bin/env node

import { createServer } from "node:http";
import { readFileSync, watchFile } from "node:fs";
import { request as httpRequest } from "node:http";
import { request as httpsRequest } from "node:https";
import { randomUUID } from "node:crypto";
import { execSync } from "node:child_process";

const PORT = parseInt(process.env.FACTORY_CURSOR_PORT || "8316", 10);
const HOST = process.env.FACTORY_CURSOR_HOST || "127.0.0.1";
const CONFIG_PATH =
  process.env.FACTORY_CONFIG || `${process.env.HOME}/.factory/config.json`;
const MODEL_PREFIX = "fx-";
const BEARER_TOKEN = process.env.BEARER_TOKEN || "change-me";

// Old prefix mappings for backward compatibility with existing Cursor configs
// Maps prefix to base_url patterns (multiple patterns per prefix supported)
const PREFIX_PATTERNS = {
  "vibeproxy/": ["http://127.0.0.1:8318", "http://127.0.0.1:8318/v1"],
  "homelab/": ["https://api.homelabai.org/v1", "https://api.homelabai.org"],
  "kimi/": ["https://api.kimi.com/coding", "https://api.kimi.com/coding/v1"],
  "minimax/": ["https://api.minimax.io/anthropic", "https://api.minimax.io/anthropic/v1"],
  "proxy/": ["https://api-proxy.homelabai.org/v1", "https://api-proxy.homelabai.org"],
  "zai-anthropic/": ["https://api.z.ai/api/anthropic", "https://api.z.ai/api/anthropic/v1"],
  "zai/": ["https://api.z.ai/api/paas/v4"],
  "cursor/": ["http://127.0.0.1:8323/v1", "http://127.0.0.1:8323"],
};

let routeTable = new Map();
let modelList = [];

// Entra ID (Azure AD) token cache: { resource -> { token, expiresOn } }
const entraTokenCache = new Map();
const ENTRA_DEFAULT_RESOURCE = "https://cognitiveservices.azure.com";

function getEntraToken(resource = ENTRA_DEFAULT_RESOURCE) {
  const cached = entraTokenCache.get(resource);
  const now = Date.now();
  // Reuse cached token if it has more than 2 minutes of validity left
  if (cached && cached.expiresOn - now > 2 * 60 * 1000) {
    return cached.token;
  }
  try {
    const raw = execSync(
      `az account get-access-token --resource ${resource} --query '{accessToken:accessToken,expiresOn:expiresOn}' -o json`,
      { encoding: "utf-8", timeout: 15000, stdio: ["pipe", "pipe", "pipe"] }
    );
    const result = JSON.parse(raw);
    const token = result.accessToken;
    const expiresOn = new Date(result.expiresOn).getTime();
    entraTokenCache.set(resource, { token, expiresOn });
    log(`Entra token acquired for ${resource} (expires ${new Date(expiresOn).toISOString()})`);
    return token;
  } catch (e) {
    log(`ERROR acquiring Entra token for ${resource}: ${e.message}`);
    // Return stale cached token as fallback if available
    if (cached) {
      log(`Using expired Entra token as fallback`);
      return cached.token;
    }
    throw new Error(`Failed to acquire Entra token: ${e.message}. Ensure 'az login' has been run.`);
  }
}

function loadConfig() {
  try {
    const raw = readFileSync(CONFIG_PATH, "utf-8");
    const config = JSON.parse(raw);
    const newRoutes = new Map();
    const newModels = [];
    for (const m of config.custom_models || []) {
      const fxName = MODEL_PREFIX + m.model;
      const provider = m.provider || "generic-chat-completion-api";
      const isAnthropic = provider === "anthropic";
      let upstreamBase = m.base_url;
      const isLocalOpenAICompatAnthropic =
        provider === "anthropic" &&
        /^http:\/\/127\.0\.0\.1:8318(?:\/v1)?\/?$/.test(upstreamBase);

      const useEntraAuth = !m.api_key;
      const route = {
        model: m.model,
        baseUrl: upstreamBase,
        apiKey: m.api_key,
        provider,
        isAnthropic: isAnthropic && !isLocalOpenAICompatAnthropic,
        isLocalOpenAICompatAnthropic,
        extraHeaders: m.extra_headers || m.headers || {},
        extraArgs: m.extra_args || {},
        supportsImages: m.supports_images !== false,
        displayName: m.model_display_name || m.model,
        useEntraAuth,
        entraResource: m.entra_resource || ENTRA_DEFAULT_RESOURCE,
      };

      // Register with fx- prefix
      newRoutes.set(fxName, route);
      
      // Also register with old prefix-based names for backward compatibility
      for (const [prefix, patterns] of Object.entries(PREFIX_PATTERNS)) {
        for (const pattern of patterns) {
          if (upstreamBase === pattern || upstreamBase.startsWith(pattern.replace(/\/v1$/, ""))) {
            const prefixedName = prefix + m.model;
            if (!newRoutes.has(prefixedName)) {
              newRoutes.set(prefixedName, { ...route, displayName: prefixedName });
              newModels.push({
                id: prefixedName,
                object: "model",
                created: Math.floor(Date.now() / 1000),
                owned_by: (m.model_display_name || "factory").split(" - ")[0].trim(),
              });
            }
            break;
          }
        }
      }

      newModels.push({
        id: fxName,
        object: "model",
        created: Math.floor(Date.now() / 1000),
        owned_by: (m.model_display_name || "factory").split(" - ")[0].trim(),
      });
    }
    routeTable = newRoutes;
    modelList = newModels;
    const entraCount = [...newRoutes.values()].filter(r => r.useEntraAuth).length;
    log(`Loaded ${newRoutes.size} model routes (${config.custom_models?.length || 0} base models${entraCount ? `, ${entraCount} using Entra auth` : ''})`);
  } catch (e) {
    log(`ERROR loading config: ${e.message}`);
  }
}

function log(msg) {
  const ts = new Date().toISOString();
  process.stderr.write(`[${ts}] ${msg}\n`);
}

function summarizeBody(body) {
  try {
    return JSON.stringify({
      model: body.model,
      stream: body.stream,
      messages: (body.messages || []).map((m, i) => ({
        i,
        role: m?.role,
        contentType: Array.isArray(m?.content) ? "array" : typeof m?.content,
        contentPreview: typeof m?.content === "string"
          ? m.content.slice(0, 120)
          : Array.isArray(m?.content)
            ? m.content.map((p) => p?.type || typeof p).slice(0, 8)
            : null,
      })),
    });
  } catch {
    return "<unserializable body>";
  }
}

function sse(data) {
  return `data: ${JSON.stringify(data)}\n\n`;
}

function makeChunk(id, model, opts = {}) {
  const delta = {};
  if (opts.role) delta.role = opts.role;
  if (opts.content !== undefined) delta.content = opts.content;
  if (opts.reasoning !== undefined) delta.reasoning_content = opts.reasoning;
  return {
    id,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      { index: 0, delta, finish_reason: opts.finishReason || null },
    ],
  };
}

function normalizeAnthropicContent(content) {
  if (typeof content === "string") {
    return content.trim() ? [{ type: "text", text: content }] : [];
  }
  if (!Array.isArray(content)) {
    return [];
  }
  const normalized = [];
  for (const part of content) {
    if (!part) continue;
    if (typeof part === "string") {
      if (part.trim()) normalized.push({ type: "text", text: part });
      continue;
    }
    if (part.type === "text" && typeof part.text === "string" && part.text.trim()) {
      normalized.push({ type: "text", text: part.text });
      continue;
    }
    if (part.type === "image_url" && part.image_url?.url) {
      normalized.push({
        type: "image",
        source: {
          type: "url",
          url: part.image_url.url,
        },
      });
      continue;
    }
    if (part.type === "input_text" && typeof part.text === "string" && part.text.trim()) {
      normalized.push({ type: "text", text: part.text });
    }
  }
  return normalized;
}

function normalizeOpenAIContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const textParts = [];
  for (const part of content) {
    if (!part) continue;
    if (typeof part === "string") {
      if (part.trim()) textParts.push(part);
      continue;
    }
    if (part.type === "text" && typeof part.text === "string" && part.text.trim()) {
      textParts.push(part.text);
      continue;
    }
    if (part.type === "input_text" && typeof part.text === "string" && part.text.trim()) {
      textParts.push(part.text);
    }
  }
  return textParts.join("\n");
}

function sanitizeOpenAIMessages(messages = []) {
  const sanitized = [];
  for (const msg of messages) {
    if (!msg || !msg.role) continue;
    const content = normalizeOpenAIContent(msg.content);
    if (!content.trim() && msg.role !== "assistant") continue;
    sanitized.push({
      role: msg.role,
      content,
    });
  }
  return sanitized;
}

// Convert OpenAI Responses API 'input' format to Chat Completions 'messages' format
function responsesInputToMessages(input, instructions) {
  const messages = [];
  // 'instructions' in Responses API maps to a system message
  if (instructions && typeof instructions === "string" && instructions.trim()) {
    messages.push({ role: "system", content: instructions });
  }
  if (typeof input === "string") {
    // Simple string input -> single user message
    messages.push({ role: "user", content: input });
    return messages;
  }
  if (!Array.isArray(input)) return messages;
  for (const item of input) {
    if (!item) continue;
    if (typeof item === "string") {
      messages.push({ role: "user", content: item });
      continue;
    }
    // Responses API message objects have 'role' and 'content'
    if (item.role && item.content !== undefined) {
      const role = item.role === "developer" ? "system" : item.role;
      let content;
      if (typeof item.content === "string") {
        content = item.content;
      } else if (Array.isArray(item.content)) {
        // Extract text from content parts
        content = item.content
          .map((p) => {
            if (typeof p === "string") return p;
            if (p?.type === "input_text" || p?.type === "text") return p.text || "";
            if (p?.type === "output_text") return p.text || "";
            return "";
          })
          .filter(Boolean)
          .join("\n");
      } else {
        content = String(item.content || "");
      }
      if (content.trim() || role === "assistant") {
        messages.push({ role, content });
      }
      continue;
    }
    // Bare content part objects (e.g. { type: "input_text", text: "..." })
    if (item.type === "input_text" && item.text) {
      messages.push({ role: "user", content: item.text });
    }
  }
  return messages;
}

// Convert Responses API tool format to Chat Completions tool format
// Only function tools are supported by /v1/chat/completions; other types are dropped.
function convertResponsesTools(tools) {
  if (!Array.isArray(tools)) return tools;
  const converted = [];
  for (const tool of tools) {
    // Already in Chat Completions format (has tool.function)
    if (tool.type === "function" && tool.function) {
      converted.push(tool);
      continue;
    }
    // Responses API format: { type: "function", name: "...", description: "...", parameters: {...} }
    if (tool.type === "function" && tool.name && !tool.function) {
      converted.push({
        type: "function",
        function: {
          name: tool.name,
          ...(tool.description !== undefined && { description: tool.description }),
          ...(tool.parameters !== undefined && { parameters: tool.parameters }),
          ...(tool.strict !== undefined && { strict: tool.strict }),
        },
      });
      continue;
    }
    // Drop non-function tools (mcp, computer_use, custom, etc.) — unsupported by Chat Completions API
  }
  return converted;
}

function openAINonStreamToSSE(parsed, fxModel, chatId, res) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });
  const content = parsed?.choices?.[0]?.message?.content || "";
  res.write(sse(makeChunk(chatId, fxModel, { role: "assistant", content: "" })));
  if (content) {
    res.write(sse(makeChunk(chatId, fxModel, { content })));
  }
  res.write(sse(makeChunk(chatId, fxModel, { finishReason: "stop" })));
  res.write("data: [DONE]\n\n");
  res.end();
}

function openaiToAnthropic(body) {
  const messages = body.messages || [];
  const systemParts = [];
  const converted = [];
  for (const msg of messages) {
    const normalizedContent = normalizeAnthropicContent(msg.content);
    if (msg.role === "system") {
      const text = normalizedContent
        .filter((p) => p.type === "text")
        .map((p) => p.text)
        .join("");
      if (text.trim()) systemParts.push(text);
      continue;
    }
    if (!["user", "assistant"].includes(msg.role)) continue;
    if (normalizedContent.length === 0) continue;
    converted.push({ role: msg.role, content: normalizedContent });
  }
  const result = {
    model: body.model,
    messages: converted,
    max_tokens: body.max_tokens || 8192,
  };
  if (systemParts.length) result.system = systemParts.join("\n");
  if (body.temperature !== undefined) result.temperature = body.temperature;
  if (body.stream) result.stream = true;
  return result;
}

function anthropicNonStreamToOpenAI(resp, model, chatId) {
  const contentParts = resp.content || [];
  let text = "";
  for (const part of contentParts) {
    if (part.type === "text") text += part.text || "";
    if (part.type === "thinking" && part.thinking) text += part.thinking;
  }
  const usage = resp.usage || {};
  return {
    id: chatId,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: text },
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: usage.input_tokens || 0,
      completion_tokens: usage.output_tokens || 0,
      total_tokens: (usage.input_tokens || 0) + (usage.output_tokens || 0),
    },
  };
}

function proxyUpstream(route, body, isStream, req, res) {
  const chatId = `chatcmpl-${randomUUID().replace(/-/g, "").slice(0, 24)}`;
  const fxModel = body.model;
  const realModel = route.model;
  body.model = realModel;
  const forceNonStreamForLocalClaude = route.isLocalOpenAICompatAnthropic && isStream;

  // Merge extra_args into the body
  if (route.extraArgs && Object.keys(route.extraArgs).length > 0) {
    Object.assign(body, route.extraArgs);
  }

  let url, headers, payload;

  // Resolve API key: use Entra token if api_key is empty
  let resolvedApiKey = route.apiKey;
  if (route.useEntraAuth) {
    try {
      resolvedApiKey = getEntraToken(route.entraResource);
    } catch (e) {
      log(`Entra auth failed for ${fxModel}: ${e.message}`);
      if (!res.headersSent) {
        res.writeHead(401, { "Content-Type": "application/json" });
      }
      res.end(JSON.stringify({ error: { message: e.message } }));
      return;
    }
  }

  if (route.isAnthropic) {
    const anthropicBody = openaiToAnthropic(body);
    let base = route.baseUrl.replace(/\/+$/, "");
    // Ensure we hit /v1/messages for Anthropic-protocol endpoints
    if (!base.endsWith("/v1")) base += "/v1";
    url = new URL(base + "/messages");
    payload = JSON.stringify(anthropicBody);
    headers = {
      "Content-Type": "application/json",
      "x-api-key": resolvedApiKey,
      "anthropic-version": "2023-06-01",
      ...route.extraHeaders,
    };
  } else {
    let base = route.baseUrl.replace(/\/+$/, "");
    // For OpenAI-compat endpoints, ensure /v1 path prefix
    if (!base.includes("/v1")) {
      url = new URL(base + "/v1/chat/completions");
    } else {
      url = new URL(base + "/chat/completions");
    }
    // Allowlist of valid /v1/chat/completions parameters — strip everything else
    const CHAT_COMPLETIONS_PARAMS = new Set([
      'model', 'messages', 'stream', 'stream_options',
      'temperature', 'top_p', 'n', 'max_tokens', 'max_completion_tokens',
      'stop', 'presence_penalty', 'frequency_penalty', 'logit_bias',
      'logprobs', 'top_logprobs', 'response_format', 'seed',
      'tools', 'tool_choice', 'parallel_tool_calls',
      'user', 'function_call', 'functions',
      'service_tier', 'modalities', 'audio', 'prediction', 'web_search_options',
    ]);
    const chatBody = {};
    const stripped = [];
    for (const [key, value] of Object.entries(body)) {
      if (CHAT_COMPLETIONS_PARAMS.has(key)) {
        chatBody[key] = value;
      } else {
        stripped.push(key);
      }
    }
    if (stripped.length) {
      log(`Stripped non-Chat-Completions params from ${fxModel}: ${stripped.join(', ')}`);
    }
    // Recover input/instructions for message conversion
    const input = body.input;
    const instructions = body.instructions;
    // Convert Responses API 'input' to Chat Completions 'messages' when messages are missing
    let messages = chatBody.messages;
    if ((!messages || messages.length === 0) && input) {
      messages = responsesInputToMessages(input, instructions);
      log(`Converted Responses API input to ${messages.length} messages for ${fxModel}`);
    }
    let openaiBody = {
      ...chatBody,
      stream: forceNonStreamForLocalClaude ? false : chatBody.stream,
      messages: sanitizeOpenAIMessages(messages || []),
    };
    // Convert Responses API tool format if needed
    if (openaiBody.tools) {
      openaiBody.tools = convertResponsesTools(openaiBody.tools);
    }
    if (route.isLocalOpenAICompatAnthropic) {
      openaiBody = {
        model: realModel,
        messages: sanitizeOpenAIMessages(body.messages),
        stream: forceNonStreamForLocalClaude ? false : !!body.stream,
      };
      if (body.temperature !== undefined) openaiBody.temperature = body.temperature;
      if (body.max_tokens !== undefined) openaiBody.max_tokens = body.max_tokens;
    }
    payload = JSON.stringify(openaiBody);
    headers = {
      "Content-Type": "application/json",
      Authorization: `Bearer ${resolvedApiKey}`,
      ...route.extraHeaders,
    };
  }

  headers["Content-Length"] = Buffer.byteLength(payload);

  const transport = url.protocol === "https:" ? httpsRequest : httpRequest;

  const upstream = transport(
    {
      hostname: url.hostname,
      port: url.port || (url.protocol === "https:" ? 443 : 80),
      path: url.pathname + url.search,
      method: "POST",
      headers,
      timeout: 300000,
    },
    (upRes) => {
      if (upRes.statusCode >= 400) {
        let errBody = "";
        upRes.on("data", (c) => (errBody += c));
        upRes.on("end", () => {
          log(
            `Upstream error ${upRes.statusCode} for ${fxModel}: ${errBody.slice(0, 500)}`
          );
          if (!res.headersSent) {
            res.writeHead(upRes.statusCode, {
              "Content-Type": "application/json",
            });
          }
          res.end(errBody);
        });
        return;
      }

      if (!isStream || forceNonStreamForLocalClaude) {
        let data = "";
        upRes.on("data", (c) => (data += c));
        upRes.on("end", () => {
          try {
            let parsed = JSON.parse(data);
            if (route.isAnthropic) {
              parsed = anthropicNonStreamToOpenAI(parsed, fxModel, chatId);
            } else if (forceNonStreamForLocalClaude) {
              parsed.model = fxModel;
              parsed.id = chatId;
              openAINonStreamToSSE(parsed, fxModel, chatId, res);
              return;
            } else {
              // Restore fx model name in response
              parsed.model = fxModel;
              parsed.id = chatId;
            }
            const out = JSON.stringify(parsed);
            res.writeHead(200, {
              "Content-Type": "application/json",
              "Content-Length": Buffer.byteLength(out),
            });
            res.end(out);
          } catch (e) {
            log(`Parse error for ${fxModel}: ${e.message}`);
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(data);
          }
        });
        return;
      }

      // Streaming
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      });
      // Send initial role chunk
      res.write(sse(makeChunk(chatId, fxModel, { role: "assistant", content: "" })));

      if (route.isAnthropic) {
        // Anthropic SSE -> OpenAI SSE translation
        let buf = "";
        upRes.on("data", (chunk) => {
          buf += chunk.toString();
          const lines = buf.split("\n");
          buf = lines.pop() || "";
          for (const line of lines) {
            if (!line.startsWith("data: ")) continue;
            const payload = line.slice(6).trim();
            if (payload === "[DONE]") {
              res.write(sse(makeChunk(chatId, fxModel, { finishReason: "stop" })));
              res.write("data: [DONE]\n\n");
              continue;
            }
            try {
              const ev = JSON.parse(payload);
              if (
                ev.type === "content_block_delta" &&
                ev.delta?.type === "text_delta"
              ) {
                res.write(
                  sse(makeChunk(chatId, fxModel, { content: ev.delta.text }))
                );
              } else if (
                ev.type === "content_block_delta" &&
                ev.delta?.type === "thinking_delta"
              ) {
                res.write(
                  sse(
                    makeChunk(chatId, fxModel, {
                      reasoning: ev.delta.thinking,
                    })
                  )
                );
              } else if (ev.type === "message_stop") {
                res.write(sse(makeChunk(chatId, fxModel, { finishReason: "stop" })));
                res.write("data: [DONE]\n\n");
              }
            } catch {
              // skip malformed
            }
          }
        });
        upRes.on("end", () => {
          if (buf.trim()) {
            // process remaining buffer
            if (buf.startsWith("data: ")) {
              const payload = buf.slice(6).trim();
              if (payload !== "[DONE]") {
                try {
                  const ev = JSON.parse(payload);
                  if (ev.type === "message_stop") {
                    res.write(sse(makeChunk(chatId, fxModel, { finishReason: "stop" })));
                    res.write("data: [DONE]\n\n");
                  }
                } catch {}
              }
            }
          }
          res.end();
        });
      } else {
        // OpenAI-compat streaming passthrough with model name rewrite
        let buf = "";
        upRes.on("data", (chunk) => {
          buf += chunk.toString();
          const lines = buf.split("\n");
          buf = lines.pop() || "";
          for (const line of lines) {
            if (!line.startsWith("data: ")) {
              if (line.trim()) res.write(line + "\n");
              continue;
            }
            const payload = line.slice(6).trim();
            if (payload === "[DONE]") {
              res.write("data: [DONE]\n\n");
              continue;
            }
            try {
              const ev = JSON.parse(payload);
              ev.model = fxModel;
              ev.id = chatId;
              res.write(sse(ev));
            } catch {
              res.write(line + "\n");
            }
          }
        });
        upRes.on("end", () => {
          if (buf.trim() && buf.startsWith("data: ")) {
            const payload = buf.slice(6).trim();
            if (payload === "[DONE]") {
              res.write("data: [DONE]\n\n");
            } else {
              try {
                const ev = JSON.parse(payload);
                ev.model = fxModel;
                ev.id = chatId;
                res.write(sse(ev));
              } catch {
                res.write(buf + "\n");
              }
            }
          }
          res.end();
        });
      }
    }
  );

  upstream.on("error", (err) => {
    log(`Upstream connection error for ${fxModel}: ${err.message}`);
    if (!res.headersSent) {
      res.writeHead(502, { "Content-Type": "application/json" });
    }
    res.end(JSON.stringify({ error: { message: `upstream error: ${err.message}` } }));
  });

  upstream.on("timeout", () => {
    log(`Upstream timeout for ${fxModel}`);
    upstream.destroy();
    if (!res.headersSent) {
      res.writeHead(504, { "Content-Type": "application/json" });
    }
    res.end(JSON.stringify({ error: { message: "upstream timeout" } }));
  });

  upstream.write(payload);
  upstream.end();
}

async function readBody(req) {
  let body = "";
  for await (const chunk of req) body += chunk;
  return body;
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://${HOST}`);

  // CORS preflight
  if (req.method === "OPTIONS") {
    res.writeHead(204, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
      "Access-Control-Allow-Headers": "*",
    });
    res.end();
    return;
  }

  // Health check (no auth required)
  if (url.pathname === "/health" && req.method === "GET") {
    const out = JSON.stringify({
      ok: true,
      models: routeTable.size,
      uptime: process.uptime(),
    });
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(out);
    return;
  }

  // Models list
  if (
    (url.pathname === "/v1/models" || url.pathname === "/models") &&
    req.method === "GET"
  ) {
    const out = JSON.stringify({ object: "list", data: modelList });
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(out);
    return;
  }

  // Chat completions
  if (
    (url.pathname === "/v1/chat/completions" ||
      url.pathname === "/chat/completions") &&
    req.method === "POST"
  ) {
    const raw = await readBody(req);
    let body;
    try {
      body = JSON.parse(raw);
    } catch {
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: { message: "invalid JSON" } }));
      return;
    }

    let requestedModel = body.model || "";
    // If model doesn't have fx- prefix, try adding it
    if (!requestedModel.startsWith(MODEL_PREFIX)) {
      const withPrefix = MODEL_PREFIX + requestedModel;
      if (routeTable.has(withPrefix)) {
        requestedModel = withPrefix;
      }
    }

    const route = routeTable.get(requestedModel);
    if (!route) {
      log(`Unknown model: ${body.model}`);
      res.writeHead(400, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          error: {
            message: `Unknown model: ${body.model}. Available: ${[...routeTable.keys()].join(", ")}`,
          },
        })
      );
      return;
    }

    const isStream = body.stream !== false;
    body.model = requestedModel; // keep fx- prefix for response model field
    log(
      `${isStream ? "STREAM" : "BLOCK"} ${requestedModel} -> ${route.displayName} (${route.baseUrl})`
    );
    if (requestedModel === "fx-claude-opus-4-6") {
      log(`Claude request summary: ${summarizeBody(body)}`);
    }
    proxyUpstream(route, body, isStream, req, res);
    return;
  }

  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: { message: `Not found: ${url.pathname}` } }));
});

// Load config and start
loadConfig();

// Watch for config changes
watchFile(CONFIG_PATH, { interval: 5000 }, () => {
  log("Config file changed, reloading...");
  loadConfig();
});

server.listen(PORT, HOST, () => {
  log(`factory-cursor-bridge listening on http://${HOST}:${PORT}`);
  log(`  ${routeTable.size} models loaded`);
  log(`  POST /v1/chat/completions - Chat completions (streaming + blocking)`);
  log(`  GET  /v1/models           - List models`);
  log(`  GET  /health              - Health check`);
  log(`  Config: ${CONFIG_PATH} (auto-reload on change)`);
});
