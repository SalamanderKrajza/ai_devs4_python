import { Hono } from 'hono';
import { chat } from './chat.js';
import { mcp } from './mcp.js';
const routes = new Hono();
routes.route('/chat', chat);
routes.route('/mcp', mcp);
export { routes };
