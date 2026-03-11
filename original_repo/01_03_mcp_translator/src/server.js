/**
 * HTTP server — exposes /api/chat and /api/translate endpoints.
 *
 * Both routes delegate to the same agent loop. The server is a thin
 * HTTP shell around the MCP-backed translation agent.
 */

import { createServer } from "http";
import { run } from "./agent.js";
import log from "./helpers/logger.js";

const parseBody = (req) =>
  new Promise((resolve, reject) => {
    let data = "";
    req.on("data", (chunk) => (data += chunk));
    req.on("end", () => {
      try {
        resolve(data ? JSON.parse(data) : {});
      } catch (e) {
        reject(new Error("Invalid JSON"));
      }
    });
    req.on("error", reject);
  });

const sendJson = (res, status, data) => {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(data));
};

const getPublicBaseUrl = ({ host, port }) => {
  const publicHost = host === "0.0.0.0" || host === "::" ? "localhost" : host;
  return `http://${publicHost}:${port}`;
};

const createRequestHandler = (getMcpContext) => async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host}`);

  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    res.writeHead(204);
    res.end();
    return;
  }

  const { mcpClient, mcpTools } = getMcpContext();

  try {
    if (url.pathname === "/api/chat" && req.method === "POST") {
      if (!mcpClient) return sendJson(res, 503, { error: "MCP client not connected" });

      const body = await parseBody(req);
      if (!body.message) return sendJson(res, 400, { error: "Message is required" });

      const result = await run(body.message, { mcpClient, mcpTools });
      return sendJson(res, 200, result);
    }

    if (url.pathname === "/api/translate" && req.method === "POST") {
      if (!mcpClient) return sendJson(res, 503, { error: "MCP client not connected" });

      const body = await parseBody(req);
      if (!body.text) return sendJson(res, 400, { error: "Text is required" });

      const query = `Translate the following text to English. Preserve tone, formatting, and nuances:\n\n${body.text}`;
      const result = await run(query, { mcpClient, mcpTools });
      return sendJson(res, 200, { translation: result.response });
    }

    sendJson(res, 404, { error: "Not found" });
  } catch (error) {
    log.error("Request error", error.message);
    sendJson(res, 500, { error: error.message });
  }
};

/**
 * Creates and starts the HTTP server.
 *
 * @param {object} config — { host, port }
 * @param {function} getMcpContext — returns { mcpClient, mcpTools }
 * @returns {import("http").Server}
 */
export const startHttpServer = (config, getMcpContext) => {
  const server = createServer(createRequestHandler(getMcpContext));

  server.listen(config.port, config.host, () => {
    const baseUrl = getPublicBaseUrl(config);

    log.ready(`Server listening on ${baseUrl}`);
    log.info("Endpoints:");
    log.endpoint("POST", "/api/chat", "Chat with agent");
    log.endpoint("POST", "/api/translate", "Translate text");
    log.info("Example curl:");
    log.info(`  curl -X POST "${baseUrl}/api/translate" -H "Content-Type: application/json" -d '{"text":"To jest przykladowy tekst po polsku."}'`);
  });

  return server;
};
