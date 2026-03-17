/**
 * OpenAI-compatible provider adapter (Responses API)
 */
import OpenAI from 'openai';
import { throwIfAborted } from '../types.js';
// ─────────────────────────────────────────────────────────────────────────────
// Input mapping (our types → OpenAI)
// ─────────────────────────────────────────────────────────────────────────────
function toOpenAIInput(items) {
    const result = [];
    for (const item of items) {
        switch (item.type) {
            case 'message':
                result.push({
                    role: item.role,
                    content: typeof item.content === 'string'
                        ? item.content
                        : item.content.map(part => {
                            if (part.type === 'text') {
                                return { type: 'input_text', text: part.text };
                            }
                            const url = 'uri' in part ? part.uri : `data:${part.mimeType};base64,${part.data}`;
                            return { type: 'input_image', image_url: url, detail: 'auto' };
                        }),
                });
                break;
            case 'function_call':
                result.push({
                    type: 'function_call',
                    call_id: item.callId,
                    name: item.name,
                    arguments: JSON.stringify(item.arguments),
                });
                break;
            case 'function_result':
                result.push({
                    type: 'function_call_output',
                    call_id: item.callId,
                    output: item.output,
                });
                break;
            case 'reasoning':
                // Skip reasoning items - OpenAI doesn't accept them as input
                // This enables multi-provider sessions where reasoning from Gemini/other
                // providers is silently skipped
                break;
        }
    }
    return result;
}
function toOpenAITools(tools, webSearchMode = 'tool') {
    if (!tools?.length)
        return undefined;
    const mappedTools = tools.flatMap((t) => {
        if (t.type === 'web_search') {
            return webSearchMode === 'tool'
                ? [{ type: 'web_search_preview' }]
                : [];
        }
        return [{
                type: 'function',
                name: t.name,
                description: t.description,
                parameters: t.parameters,
                strict: false,
            }];
    });
    return mappedTools.length > 0 ? mappedTools : undefined;
}
// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────
function safeParseJson(json) {
    try {
        return JSON.parse(json);
    }
    catch {
        return {};
    }
}
// ─────────────────────────────────────────────────────────────────────────────
// Output mapping (OpenAI → our types)
// ─────────────────────────────────────────────────────────────────────────────
function fromOpenAIOutput(items) {
    const result = [];
    for (const item of items) {
        if (item.type === 'message') {
            const text = item.content
                .filter((c) => c.type === 'output_text')
                .map(c => c.text)
                .join('');
            if (text)
                result.push({ type: 'text', text });
        }
        if (item.type === 'function_call') {
            result.push({
                type: 'function_call',
                callId: item.call_id,
                name: item.name,
                arguments: safeParseJson(item.arguments),
            });
        }
        if (item.type === 'reasoning') {
            const text = item.summary
                .filter((s) => s.type === 'summary_text')
                .map(s => s.text)
                .join('');
            // Tag provider for multi-provider session support
            // OpenAI doesn't use signatures - reasoning is display-only
            if (text)
                result.push({ type: 'reasoning', text, provider: 'openai' });
        }
    }
    return result;
}
function hasToolCalls(output) {
    return output.some(o => o.type === 'function_call');
}
function createStreamState() {
    return { output: [], fnCallMeta: new Map() };
}
function accumulate(state, event) {
    switch (event.type) {
        case 'response.output_item.added':
            state.output[event.output_index] = event.item;
            if (event.item.type === 'function_call') {
                state.fnCallMeta.set(event.output_index, {
                    callId: event.item.call_id,
                    name: event.item.name,
                });
            }
            break;
        case 'response.content_part.added': {
            const output = state.output[event.output_index];
            if (output?.type === 'message') {
                output.content[event.content_index] = event.part;
            }
            break;
        }
        case 'response.output_text.delta': {
            const output = state.output[event.output_index];
            if (output?.type === 'message') {
                const content = output.content[event.content_index];
                if (content?.type === 'output_text') {
                    content.text += event.delta;
                }
            }
            break;
        }
        case 'response.function_call_arguments.delta': {
            const output = state.output[event.output_index];
            if (output?.type === 'function_call') {
                output.arguments += event.delta;
            }
            break;
        }
        case 'response.reasoning_summary_text.delta': {
            const output = state.output[event.output_index];
            if (output?.type === 'reasoning') {
                const summary = output.summary[event.summary_index];
                if (summary?.type === 'summary_text') {
                    summary.text += event.delta;
                }
            }
            break;
        }
    }
}
// ─────────────────────────────────────────────────────────────────────────────
// Provider implementation
// ─────────────────────────────────────────────────────────────────────────────
export function createOpenAIProvider(config) {
    const client = new OpenAI({
        apiKey: config.apiKey,
        baseURL: config.baseUrl,
        defaultHeaders: config.defaultHeaders,
    });
    return {
        name: config.name ?? 'openai',
        async generate(request) {
            throwIfAborted(request.signal);
            const model = config.transformModel?.(request.model, request) ?? request.model;
            const response = await client.responses.create({
                model,
                instructions: request.instructions,
                input: toOpenAIInput(request.input),
                tools: toOpenAITools(request.tools, config.webSearchMode),
                temperature: request.temperature,
                max_output_tokens: request.maxTokens,
            });
            return {
                id: response.id,
                model: response.model,
                output: fromOpenAIOutput(response.output),
                finishReason: hasToolCalls(response.output) ? 'tool_calls' : 'stop',
                usage: response.usage && {
                    inputTokens: response.usage.input_tokens,
                    outputTokens: response.usage.output_tokens,
                    totalTokens: response.usage.total_tokens,
                    cachedTokens: response.usage.input_tokens_details?.cached_tokens ?? 0,
                },
            };
        },
        async *stream(request) {
            throwIfAborted(request.signal);
            const model = config.transformModel?.(request.model, request) ?? request.model;
            const rawStream = await client.responses.create({
                model,
                instructions: request.instructions,
                input: toOpenAIInput(request.input),
                tools: toOpenAITools(request.tools, config.webSearchMode),
                temperature: request.temperature,
                max_output_tokens: request.maxTokens,
                stream: true,
            });
            const state = createStreamState();
            for await (const event of rawStream) {
                // Accumulate state first (like SDK does)
                accumulate(state, event);
                // Then emit our normalized events
                switch (event.type) {
                    case 'response.output_text.delta':
                        yield { type: 'text_delta', delta: event.delta };
                        break;
                    case 'response.output_text.done':
                        yield { type: 'text_done', text: event.text };
                        break;
                    case 'response.function_call_arguments.delta': {
                        const meta = state.fnCallMeta.get(event.output_index);
                        if (meta) {
                            yield {
                                type: 'function_call_delta',
                                callId: meta.callId,
                                name: meta.name,
                                argumentsDelta: event.delta,
                            };
                        }
                        break;
                    }
                    case 'response.function_call_arguments.done': {
                        const meta = state.fnCallMeta.get(event.output_index);
                        if (meta) {
                            yield {
                                type: 'function_call_done',
                                callId: meta.callId,
                                name: meta.name,
                                arguments: safeParseJson(event.arguments),
                            };
                        }
                        break;
                    }
                    case 'response.reasoning_summary_text.delta':
                        yield { type: 'reasoning_delta', delta: event.delta };
                        break;
                    case 'response.reasoning_summary_text.done':
                        yield { type: 'reasoning_done', text: event.text };
                        break;
                    case 'response.failed':
                        yield {
                            type: 'error',
                            error: event.response.error?.message ?? 'Response failed',
                            code: event.response.error?.code,
                        };
                        break;
                    case 'response.completed':
                        yield {
                            type: 'done',
                            response: {
                                id: event.response.id,
                                model: event.response.model,
                                output: fromOpenAIOutput(event.response.output),
                                finishReason: hasToolCalls(event.response.output) ? 'tool_calls' : 'stop',
                                usage: event.response.usage && {
                                    inputTokens: event.response.usage.input_tokens,
                                    outputTokens: event.response.usage.output_tokens,
                                    totalTokens: event.response.usage.total_tokens,
                                    cachedTokens: event.response.usage.input_tokens_details?.cached_tokens ?? 0,
                                },
                            },
                        };
                        break;
                }
            }
        },
    };
}
