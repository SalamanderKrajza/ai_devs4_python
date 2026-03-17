Create a model response
post
 
https://api.openai.com/v1/responses
Creates a model response. Provide text or image inputs to generate text or JSON outputs. Have the model call your own custom code or use built-in tools like web search or file search to use your own data as input for the model's response.

Request body
background
boolean

Optional
Defaults to false
Whether to run the model response in the background. Learn more.

conversation
string or object

Optional
Defaults to null
The conversation that this response belongs to. Items from this conversation are prepended to input_items for this response request. Input items and output items from this response are automatically added to this conversation after this response completes.


Show possible types
include
array

Optional
Specify additional output data to include in the model response. Currently supported values are:

web_search_call.action.sources: Include the sources of the web search tool call.
code_interpreter_call.outputs: Includes the outputs of python code execution in code interpreter tool call items.
computer_call_output.output.image_url: Include image urls from the computer call output.
file_search_call.results: Include the search results of the file search tool call.
message.input_image.image_url: Include image urls from the input message.
message.output_text.logprobs: Include logprobs with assistant messages.
reasoning.encrypted_content: Includes an encrypted version of reasoning tokens in reasoning item outputs. This enables reasoning items to be used in multi-turn conversations when using the Responses API statelessly (like when the store parameter is set to false, or when an organization is enrolled in the zero data retention program).
input
string or array

Optional
Text, image, or file inputs to the model, used to generate a response.

Learn more:

Text inputs and outputs
Image inputs
File inputs
Conversation state
Function calling

Show possible types
instructions
string

Optional
A system (or developer) message inserted into the model's context.

When using along with previous_response_id, the instructions from a previous response will not be carried over to the next response. This makes it simple to swap out system (or developer) messages in new responses.

max_output_tokens
integer

Optional
An upper bound for the number of tokens that can be generated for a response, including visible output tokens and reasoning tokens.

max_tool_calls
integer

Optional
The maximum number of total calls to built-in tools that can be processed in a response. This maximum number applies across all built-in tool calls, not per individual tool. Any further attempts to call a tool by the model will be ignored.

metadata
map

Optional
Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard.

Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters.

model
string

Optional
Model ID used to generate the response, like gpt-4o or o3. OpenAI offers a wide range of models with different capabilities, performance characteristics, and price points. Refer to the model guide to browse and compare available models.

parallel_tool_calls
boolean

Optional
Defaults to true
Whether to allow the model to run tool calls in parallel.

previous_response_id
string

Optional
The unique ID of the previous response to the model. Use this to create multi-turn conversations. Learn more about conversation state. Cannot be used in conjunction with conversation.

prompt
object

Optional
Reference to a prompt template and its variables. Learn more.


Show properties
prompt_cache_key
string

Optional
Used by OpenAI to cache responses for similar requests to optimize your cache hit rates. Replaces the user field. Learn more.

prompt_cache_retention
string

Optional
The retention policy for the prompt cache. Set to 24h to enable extended prompt caching, which keeps cached prefixes active for longer, up to a maximum of 24 hours. Learn more.

reasoning
object

Optional
gpt-5 and o-series models only

Configuration options for reasoning models.


Show properties
safety_identifier
string

Optional
A stable identifier used to help detect users of your application that may be violating OpenAI's usage policies. The IDs should be a string that uniquely identifies each user. We recommend hashing their username or email address, in order to avoid sending us any identifying information. Learn more.

service_tier
string

Optional
Defaults to auto
Specifies the processing type used for serving the request.

If set to 'auto', then the request will be processed with the service tier configured in the Project settings. Unless otherwise configured, the Project will use 'default'.
If set to 'default', then the request will be processed with the standard pricing and performance for the selected model.
If set to 'flex' or 'priority', then the request will be processed with the corresponding service tier.
When not set, the default behavior is 'auto'.
When the service_tier parameter is set, the response body will include the service_tier value based on the processing mode actually used to serve the request. This response value may be different from the value set in the parameter.

store
boolean

Optional
Defaults to true
Whether to store the generated model response for later retrieval via API.

stream
boolean

Optional
Defaults to false
If set to true, the model response data will be streamed to the client as it is generated using server-sent events. See the Streaming section below for more information.

stream_options
object

Optional
Defaults to null
Options for streaming responses. Only set this when you set stream: true.


Show properties
temperature
number

Optional
Defaults to 1
What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top_p but not both.

text
object

Optional
Configuration options for a text response from the model. Can be plain text or structured JSON data. Learn more:

Text inputs and outputs
Structured Outputs

Show properties
tool_choice
string or object

Optional
How the model should select which tool (or tools) to use when generating a response. See the tools parameter to see how to specify which tools the model can call.


Show possible types
tools
array

Optional
An array of tools the model may call while generating a response. You can specify which tool to use by setting the tool_choice parameter.

We support the following categories of tools:

Built-in tools: Tools that are provided by OpenAI that extend the model's capabilities, like web search or file search. Learn more about built-in tools.
MCP Tools: Integrations with third-party systems via custom MCP servers or predefined connectors such as Google Drive and SharePoint. Learn more about MCP Tools.
Function calls (custom tools): Functions that are defined by you, enabling the model to call your own code with strongly typed arguments and outputs. Learn more about function calling. You can also use custom tools to call your own code.

Show possible types
top_logprobs
integer

Optional
An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability.

top_p
number

Optional
Defaults to 1
An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.

We generally recommend altering this or temperature but not both.

truncation
string

Optional
Defaults to disabled
The truncation strategy to use for the model response.

auto: If the input to this Response exceeds the model's context window size, the model will truncate the response to fit the context window by dropping items from the beginning of the conversation.
disabled (default): If the input size will exceed the context window size for a model, the request will fail with a 400 error.
user
Deprecated
string

Optional
This field is being replaced by safety_identifier and prompt_cache_key. Use prompt_cache_key instead to maintain caching optimizations. A stable identifier for your end-users. Used to boost cache hit rates by better bucketing similar requests and to help OpenAI detect and prevent abuse. Learn more.

Returns
Returns a Response object.

Text input
Image input
File input
Web search
File search
Streaming
Functions
Reasoning
Example request
curl https://api.openai.com/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4.1",
    "input": "Tell me a three sentence bedtime story about a unicorn."
  }'
Response
{
  "id": "resp_67ccd2bed1ec8190b14f964abc0542670bb6a6b452d3795b",
  "object": "response",
  "created_at": 1741476542,
  "status": "completed",
  "completed_at": 1741476543,
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": null,
  "model": "gpt-4.1-2025-04-14",
  "output": [
    {
      "type": "message",
      "id": "msg_67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "In a peaceful grove beneath a silver moon, a unicorn named Lumina discovered a hidden pool that reflected the stars. As she dipped her horn into the water, the pool began to shimmer, revealing a pathway to a magical realm of endless night skies. Filled with wonder, Lumina whispered a wish for all who dream to find their own hidden magic, and as she glanced back, her hoofprints sparkled like stardust.",
          "annotations": []
        }
      ]
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {
    "effort": null,
    "summary": null
  },
  "store": true,
  "temperature": 1.0,
  "text": {
    "format": {
      "type": "text"
    }
  },
  "tool_choice": "auto",
  "tools": [],
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 36,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 87,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 123
  },
  "user": null,
  "metadata": {}
}
Get a model response
get
 
https://api.openai.com/v1/responses/{response_id}
Retrieves a model response with the given ID.

Path parameters
response_id
string

Required
The ID of the response to retrieve.

Query parameters
include
array

Optional
Additional fields to include in the response. See the include parameter for Response creation above for more information.

include_obfuscation
boolean

Optional
When true, stream obfuscation will be enabled. Stream obfuscation adds random characters to an obfuscation field on streaming delta events to normalize payload sizes as a mitigation to certain side-channel attacks. These obfuscation fields are included by default, but add a small amount of overhead to the data stream. You can set include_obfuscation to false to optimize for bandwidth if you trust the network links between your application and the OpenAI API.

starting_after
integer

Optional
The sequence number of the event after which to start streaming.

stream
boolean

Optional
If set to true, the model response data will be streamed to the client as it is generated using server-sent events. See the Streaming section below for more information.

Returns
The Response object matching the specified ID.

Example request
curl https://api.openai.com/v1/responses/resp_123 \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY"
Response
{
  "id": "resp_67cb71b351908190a308f3859487620d06981a8637e6bc44",
  "object": "response",
  "created_at": 1741386163,
  "status": "completed",
  "completed_at": 1741386164,
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": null,
  "model": "gpt-4o-2024-08-06",
  "output": [
    {
      "type": "message",
      "id": "msg_67cb71b3c2b0819084d481baaaf148f206981a8637e6bc44",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Silent circuits hum,  \nThoughts emerge in data streams—  \nDigital dawn breaks.",
          "annotations": []
        }
      ]
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {
    "effort": null,
    "summary": null
  },
  "store": true,
  "temperature": 1.0,
  "text": {
    "format": {
      "type": "text"
    }
  },
  "tool_choice": "auto",
  "tools": [],
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 32,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 18,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 50
  },
  "user": null,
  "metadata": {}
}curl https://api.openai.com/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4.1",
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "what is in this image?"},
          {
            "type": "input_image",
            "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
          }
        ]
      }
    ]
  }'
curl https://api.openai.com/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4.1",
    "tools": [{ "type": "web_search_preview" }],
    "input": "What was a positive news story from today?"
  }'

post
 
https://api.openai.com/v1/responses
Creates a model response. Provide text or image inputs to generate text or JSON outputs. Have the model call your own custom code or use built-in tools like web search or file search to use your own data as input for the model's response.

Request body
background
boolean

Optional
Defaults to false
Whether to run the model response in the background. Learn more.

conversation
string or object

Optional
Defaults to null
The conversation that this response belongs to. Items from this conversation are prepended to input_items for this response request. Input items and output items from this response are automatically added to this conversation after this response completes.


Show possible types
include
array

Optional
Specify additional output data to include in the model response. Currently supported values are:

web_search_call.action.sources: Include the sources of the web search tool call.
code_interpreter_call.outputs: Includes the outputs of python code execution in code interpreter tool call items.
computer_call_output.output.image_url: Include image urls from the computer call output.
file_search_call.results: Include the search results of the file search tool call.
message.input_image.image_url: Include image urls from the input message.
message.output_text.logprobs: Include logprobs with assistant messages.
reasoning.encrypted_content: Includes an encrypted version of reasoning tokens in reasoning item outputs. This enables reasoning items to be used in multi-turn conversations when using the Responses API statelessly (like when the store parameter is set to false, or when an organization is enrolled in the zero data retention program).
input
string or array

Optional
Text, image, or file inputs to the model, used to generate a response.

Learn more:

Text inputs and outputs
Image inputs
File inputs
Conversation state
Function calling

Show possible types
instructions
string

Optional
A system (or developer) message inserted into the model's context.

When using along with previous_response_id, the instructions from a previous response will not be carried over to the next response. This makes it simple to swap out system (or developer) messages in new responses.

max_output_tokens
integer

Optional
An upper bound for the number of tokens that can be generated for a response, including visible output tokens and reasoning tokens.

max_tool_calls
integer

Optional
The maximum number of total calls to built-in tools that can be processed in a response. This maximum number applies across all built-in tool calls, not per individual tool. Any further attempts to call a tool by the model will be ignored.

metadata
map

Optional
Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard.

Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters.

model
string

Optional
Model ID used to generate the response, like gpt-4o or o3. OpenAI offers a wide range of models with different capabilities, performance characteristics, and price points. Refer to the model guide to browse and compare available models.

parallel_tool_calls
boolean

Optional
Defaults to true
Whether to allow the model to run tool calls in parallel.

previous_response_id
string

Optional
The unique ID of the previous response to the model. Use this to create multi-turn conversations. Learn more about conversation state. Cannot be used in conjunction with conversation.

prompt
object

Optional
Reference to a prompt template and its variables. Learn more.


Show properties
prompt_cache_key
string

Optional
Used by OpenAI to cache responses for similar requests to optimize your cache hit rates. Replaces the user field. Learn more.

prompt_cache_retention
string

Optional
The retention policy for the prompt cache. Set to 24h to enable extended prompt caching, which keeps cached prefixes active for longer, up to a maximum of 24 hours. Learn more.

reasoning
object

Optional
gpt-5 and o-series models only

Configuration options for reasoning models.


Show properties
safety_identifier
string

Optional
A stable identifier used to help detect users of your application that may be violating OpenAI's usage policies. The IDs should be a string that uniquely identifies each user. We recommend hashing their username or email address, in order to avoid sending us any identifying information. Learn more.

service_tier
string

Optional
Defaults to auto
Specifies the processing type used for serving the request.

If set to 'auto', then the request will be processed with the service tier configured in the Project settings. Unless otherwise configured, the Project will use 'default'.
If set to 'default', then the request will be processed with the standard pricing and performance for the selected model.
If set to 'flex' or 'priority', then the request will be processed with the corresponding service tier.
When not set, the default behavior is 'auto'.
When the service_tier parameter is set, the response body will include the service_tier value based on the processing mode actually used to serve the request. This response value may be different from the value set in the parameter.

store
boolean

Optional
Defaults to true
Whether to store the generated model response for later retrieval via API.

stream
boolean

Optional
Defaults to false
If set to true, the model response data will be streamed to the client as it is generated using server-sent events. See the Streaming section below for more information.

stream_options
object

Optional
Defaults to null
Options for streaming responses. Only set this when you set stream: true.


Show properties
temperature
number

Optional
Defaults to 1
What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top_p but not both.

text
object

Optional
Configuration options for a text response from the model. Can be plain text or structured JSON data. Learn more:

Text inputs and outputs
Structured Outputs

Show properties
tool_choice
string or object

Optional
How the model should select which tool (or tools) to use when generating a response. See the tools parameter to see how to specify which tools the model can call.


Show possible types
tools
array

Optional
An array of tools the model may call while generating a response. You can specify which tool to use by setting the tool_choice parameter.

We support the following categories of tools:

Built-in tools: Tools that are provided by OpenAI that extend the model's capabilities, like web search or file search. Learn more about built-in tools.
MCP Tools: Integrations with third-party systems via custom MCP servers or predefined connectors such as Google Drive and SharePoint. Learn more about MCP Tools.
Function calls (custom tools): Functions that are defined by you, enabling the model to call your own code with strongly typed arguments and outputs. Learn more about function calling. You can also use custom tools to call your own code.

Show possible types
top_logprobs
integer

Optional
An integer between 0 and 20 specifying the number of most likely tokens to return at each token position, each with an associated log probability.

top_p
number

Optional
Defaults to 1
An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.

We generally recommend altering this or temperature but not both.

truncation
string

Optional
Defaults to disabled
The truncation strategy to use for the model response.

auto: If the input to this Response exceeds the model's context window size, the model will truncate the response to fit the context window by dropping items from the beginning of the conversation.
disabled (default): If the input size will exceed the context window size for a model, the request will fail with a 400 error.
user
Deprecated
string

Optional
This field is being replaced by safety_identifier and prompt_cache_key. Use prompt_cache_key instead to maintain caching optimizations. A stable identifier for your end-users. Used to boost cache hit rates by better bucketing similar requests and to help OpenAI detect and prevent abuse. Learn more.

Returns
Returns a Response object.

Text input
Image input
File input
Web search
File search
Streaming
Functions
Reasoning
Example request
curl https://api.openai.com/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-4.1",
    "instructions": "You are a helopenai
TypeScript icon, indicating that this package has built-in type declarations
6.17.0 • Public • Published 8 days ago
OpenAI TypeScript and JavaScript API Library
NPM version npm bundle size JSR Version

This library provides convenient access to the OpenAI REST API from TypeScript or JavaScript.

It is generated from our OpenAPI specification with Stainless.

To learn how to use the OpenAI API, check out our API Reference and Documentation.

Installation
npm install openai
Installation from JSR
deno add jsr:@openai/openai
npx jsr add @openai/openai
These commands will make the module importable from the @openai/openai scope. You can also import directly from JSR without an install step if you're using the Deno JavaScript runtime:

import OpenAI from 'jsr:@openai/openai';
Usage
The full API of this library can be found in api.md file along with many code examples.

The primary API for interacting with OpenAI models is the Responses API. You can generate text from the model with the code below.

import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: process.env['OPENAI_API_KEY'], // This is the default and can be omitted
});

const response = await client.responses.create({
  model: 'gpt-5.2',
  instructions: 'You are a coding assistant that talks like a pirate',
  input: 'Are semicolons optional in JavaScript?',
});

console.log(response.output_text);
The previous standard (supported indefinitely) for generating text is the Chat Completions API. You can use that API to generate text from the model with the code below.

import OpenAI from 'openai';

const client = new OpenAI({
  apiKey: process.env['OPENAI_API_KEY'], // This is the default and can be omitted
});

const completion = await client.chat.completions.create({
  model: 'gpt-5.2',
  messages: [
    { role: 'developer', content: 'Talk like a pirate.' },
    { role: 'user', content: 'Are semicolons optional in JavaScript?' },
  ],
});

console.log(completion.choices[0].message.content);
Streaming responses
We provide support for streaming responses using Server Sent Events (SSE).

import OpenAI from 'openai';

const client = new OpenAI();

const stream = await client.responses.create({
  model: 'gpt-5.2',
  input: 'Say "Sheep sleep deep" ten times fast!',
  stream: true,
});

for await (const event of stream) {
  console.log(event);
}
File uploads
Request parameters that correspond to file uploads can be passed in many different forms:

File (or an object with the same structure)
a fetch Response (or an object with the same structure)
an fs.ReadStream
the return value of our toFile helper
import fs from 'fs';
import OpenAI, { toFile } from 'openai';

const client = new OpenAI();

// If you have access to Node `fs` we recommend using `fs.createReadStream()`:
await client.files.create({ file: fs.createReadStream('input.jsonl'), purpose: 'fine-tune' });

// Or if you have the web `File` API you can pass a `File` instance:
await client.files.create({ file: new File(['my bytes'], 'input.jsonl'), purpose: 'fine-tune' });

// You can also pass a `fetch` `Response`:
await client.files.create({
  file: await fetch('https://somesite/input.jsonl'),
  purpose: 'fine-tune',
});

// Finally, if none of the above are convenient, you can use our `toFile` helper:
await client.files.create({
  file: await toFile(Buffer.from('my bytes'), 'input.jsonl'),
  purpose: 'fine-tune',
});
await client.files.create({
  file: await toFile(new Uint8Array([0, 1, 2]), 'input.jsonl'),
  purpose: 'fine-tune',
});
Webhook Verification
Verifying webhook signatures is optional but encouraged.

For more information about webhooks, see the API docs.

Parsing webhook payloads
For most use cases, you will likely want to verify the webhook and parse the payload at the same time. To achieve this, we provide the method client.webhooks.unwrap(), which parses a webhook request and verifies that it was sent by OpenAI. This method will throw an error if the signature is invalid.

Note that the body parameter must be the raw JSON string sent from the server (do not parse it first). The .unwrap() method will parse this JSON for you into an event object after verifying the webhook was sent from OpenAI.

import { headers } from 'next/headers';
import OpenAI from 'openai';

const client = new OpenAI({
  webhookSecret: process.env.OPENAI_WEBHOOK_SECRET, // env var used by default; explicit here.
});

export async function webhook(request: Request) {
  const headersList = headers();
  const body = await request.text();

  try {
    const event = client.webhooks.unwrap(body, headersList);

    switch (event.type) {
      case 'response.completed':
        console.log('Response completed:', event.data);
        break;
      case 'response.failed':
        console.log('Response failed:', event.data);
        break;
      default:
        console.log('Unhandled event type:', event.type);
    }

    return Response.json({ message: 'ok' });
  } catch (error) {
    console.error('Invalid webhook signature:', error);
    return new Response('Invalid signature', { status: 400 });
  }
}
Verifying webhook payloads directly
In some cases, you may want to verify the webhook separately from parsing the payload. If you prefer to handle these steps separately, we provide the method client.webhooks.verifySignature() to only verify the signature of a webhook request. Like .unwrap(), this method will throw an error if the signature is invalid.

Note that the body parameter must be the raw JSON string sent from the server (do not parse it first). You will then need to parse the body after verifying the signature.

import { headers } from 'next/headers';
import OpenAI from 'openai';

const client = new OpenAI({
  webhookSecret: process.env.OPENAI_WEBHOOK_SECRET, // env var used by default; explicit here.
});

export async function webhook(request: Request) {
  const headersList = headers();
  const body = await request.text();

  try {
    client.webhooks.verifySignature(body, headersList);

    // Parse the body after verification
    const event = JSON.parse(body);
    console.log('Verified event:', event);

    return Response.json({ message: 'ok' });
  } catch (error) {
    console.error('Invalid webhook signature:', error);
    return new Response('Invalid signature', { status: 400 });
  }
}
Handling errors
When the library is unable to connect to the API, or if the API returns a non-success status code (i.e., 4xx or 5xx response), a subclass of APIError will be thrown:

const job = await client.fineTuning.jobs
  .create({ model: 'gpt-4o', training_file: 'file-abc123' })
  .catch(async (err) => {
    if (err instanceof OpenAI.APIError) {
      console.log(err.request_id);
      console.log(err.status); // 400
      console.log(err.name); // BadRequestError
      console.log(err.headers); // {server: 'nginx', ...}
    } else {
      throw err;
    }
  });
Error codes are as follows:

Status Code	Error Type
400	BadRequestError
401	AuthenticationError
403	PermissionDeniedError
404	NotFoundError
422	UnprocessableEntityError
429	RateLimitError
>=500	InternalServerError
N/A	APIConnectionError
Request IDs
For more information on debugging requests, see these docs

All object responses in the SDK provide a _request_id property which is added from the x-request-id response header so that you can quickly log failing requests and report them back to OpenAI.

const completion = await client.chat.completions.create({
  messages: [{ role: 'user', content: 'Say this is a test' }],
  model: 'gpt-5.2',
});
console.log(completion._request_id); // req_123
You can also access the Request ID using the .withResponse() method:

const { data: stream, request_id } = await openai.chat.completions
  .create({
    model: 'gpt-5.2',
    messages: [{ role: 'user', content: 'Say this is a test' }],
    stream: true,
  })
  .withResponse();
Realtime API
The Realtime API enables you to build low-latency, multi-modal conversational experiences. It currently supports text and audio as both input and output, as well as function calling through a WebSocket connection.

import { OpenAIRealtimeWebSocket } from 'openai/realtime/websocket';

const rt = new OpenAIRealtimeWebSocket({ model: 'gpt-realtime' });

rt.on('response.text.delta', (event) => process.stdout.write(event.delta));
For more information see realtime.md.

Microsoft Azure OpenAI
To use this library with Azure OpenAI, use the AzureOpenAI class instead of the OpenAI class.

[!IMPORTANT] The Azure API shape slightly differs from the core API shape which means that the static types for responses / params won't always be correct.

import { AzureOpenAI } from 'openai';
import { getBearerTokenProvider, DefaultAzureCredential } from '@azure/identity';

const credential = new DefaultAzureCredential();
const scope = 'https://cognitiveservices.azure.com/.default';
const azureADTokenProvider = getBearerTokenProvider(credential, scope);

const openai = new AzureOpenAI({ azureADTokenProvider });

const result = await openai.chat.completions.create({
  model: 'gpt-5.2',
  messages: [{ role: 'user', content: 'Say hello!' }],
});

console.log(result.choices[0]!.message?.content);
Retries
Certain errors will be automatically retried 2 times by default, with a short exponential backoff. Connection errors (for example, due to a network connectivity problem), 408 Request Timeout, 409 Conflict, 429 Rate Limit, and >=500 Internal errors will all be retried by default.

You can use the maxRetries option to configure or disable this:

// Configure the default for all requests:
const client = new OpenAI({
  maxRetries: 0, // default is 2
});

// Or, configure per-request:
await client.chat.completions.create({ messages: [{ role: 'user', content: 'How can I get the name of the current day in JavaScript?' }], model: 'gpt-5.2' }, {
  maxRetries: 5,
});
Timeouts
Requests time out after 10 minutes by default. You can configure this with a timeout option:

// Configure the default for all requests:
const client = new OpenAI({
  timeout: 20 * 1000, // 20 seconds (default is 10 minutes)
});

// Override per-request:
await client.chat.completions.create({ messages: [{ role: 'user', content: 'How can I list all files in a directory using Python?' }], model: 'gpt-5.2' }, {
  timeout: 5 * 1000,
});
On timeout, an APIConnectionTimeoutError is thrown.

Note that requests which time out will be retried twice by default.

Request IDs
For more information on debugging requests, see these docs

All object responses in the SDK provide a _request_id property which is added from the x-request-id response header so that you can quickly log failing requests and report them back to OpenAI.

const response = await client.responses.create({ model: 'gpt-5.2', input: 'testing 123' });
console.log(response._request_id); // req_123
You can also access the Request ID using the .withResponse() method:

const { data: stream, request_id } = await openai.responses
  .create({
    model: 'gpt-5.2',
    input: 'Say this is a test',
    stream: true,
  })
  .withResponse();
Auto-pagination
List methods in the OpenAI API are paginated. You can use the for await … of syntax to iterate through items across all pages:

async function fetchAllFineTuningJobs(params) {
  const allFineTuningJobs = [];
  // Automatically fetches more pages as needed.
  for await (const fineTuningJob of client.fineTuning.jobs.list({ limit: 20 })) {
    allFineTuningJobs.push(fineTuningJob);
  }
  return allFineTuningJobs;
}
Alternatively, you can request a single page at a time:

let page = await client.fineTuning.jobs.list({ limit: 20 });
for (const fineTuningJob of page.data) {
  console.log(fineTuningJob);
}

// Convenience methods are provided for manually paginating:
while (page.hasNextPage()) {
  page = await page.getNextPage();
  // ...
}
Realtime API
The Realtime API enables you to build low-latency, multi-modal conversational experiences. It currently supports text and audio as both input and output, as well as function calling through a WebSocket connection.

import { OpenAIRealtimeWebSocket } from 'openai/realtime/websocket';

const rt = new OpenAIRealtimeWebSocket({ model: 'gpt-realtime' });

rt.on('response.text.delta', (event) => process.stdout.write(event.delta));
For more information see realtime.md.

Microsoft Azure OpenAI
To use this library with Azure OpenAI, use the AzureOpenAI class instead of the OpenAI class.

[!IMPORTANT] The Azure API shape slightly differs from the core API shape which means that the static types for responses / params won't always be correct.

import { AzureOpenAI } from 'openai';
import { getBearerTokenProvider, DefaultAzureCredential } from '@azure/identity';

const credential = new DefaultAzureCredential();
const scope = 'https://cognitiveservices.azure.com/.default';
const azureADTokenProvider = getBearerTokenProvider(credential, scope);

const openai = new AzureOpenAI({
  azureADTokenProvider,
  apiVersion: '<The API version, e.g. 2024-10-01-preview>',
});

const result = await openai.chat.completions.create({
  model: 'gpt-5.2',
  messages: [{ role: 'user', content: 'Say hello!' }],
});

console.log(result.choices[0]!.message?.content);
For more information on support for the Azure API, see azure.md.

Advanced Usage
Accessing raw Response data (e.g., headers)
The "raw" Response returned by fetch() can be accessed through the .asResponse() method on the APIPromise type that all methods return. This method returns as soon as the headers for a successful response are received and does not consume the response body, so you are free to write custom parsing or streaming logic.

You can also use the .withResponse() method to get the raw Response along with the parsed data. Unlike .asResponse() this method consumes the body, returning once it is parsed.

const client = new OpenAI();

const httpResponse = await client.responses
  .create({ model: 'gpt-5.2', input: 'say this is a test.' })
  .asResponse();

// access the underlying web standard Response object
console.log(httpResponse.headers.get('X-My-Header'));
console.log(httpResponse.statusText);

const { data: modelResponse, response: raw } = await client.responses
  .create({ model: 'gpt-5.2', input: 'say this is a test.' })
  .withResponse();
console.log(raw.headers.get('X-My-Header'));
console.log(modelResponse);
Logging
[!IMPORTANT] All log messages are intended for debugging only. The format and content of log messages may change between releases.

Log levels
The log level can be configured in two ways:

Via the OPENAI_LOG environment variable
Using the logLevel client option (overrides the environment variable if set)
import OpenAI from 'openai';

const client = new OpenAI({
  logLevel: 'debug', // Show all log messages
});
Available log levels, from most to least verbose:

'debug' - Show debug messages, info, warnings, and errors
'info' - Show info messages, warnings, and errors
'warn' - Show warnings and errors (default)
'error' - Show only errors
'off' - Disable all logging
At the 'debug' level, all HTTP requests and responses are logged, including headers and bodies. Some authentication-related headers are redacted, but sensitive data in request and response bodies may still be visible.

Custom logger
By default, this library logs to globalThis.console. You can also provide a custom logger. Most logging libraries are supported, including pino, winston, bunyan, consola, signale, and @std/log. If your logger doesn't work, please open an issue.

When providing a custom logger, the logLevel option still controls which messages are emitted, messages below the configured level will not be sent to your logger.

import OpenAI from 'openai';
import pino from 'pino';

const logger = pino();

const client = new OpenAI({
  logger: logger.child({ name: 'OpenAI' }),
  logLevel: 'debug', // Send all messages to pino, allowing it to filter
});
Making custom/undocumented requests
This library is typed for convenient access to the documented API. If you need to access undocumented endpoints, params, or response properties, the library can still be used.

Undocumented endpoints
To make requests to undocumented endpoints, you can use client.get, client.post, and other HTTP verbs. Options on the client, such as retries, will be respected when making these requests.

await client.post('/some/path', {
  body: { some_prop: 'foo' },
  query: { some_query_arg: 'bar' },
});
Undocumented request params
To make requests using undocumented parameters, you may use // @ts-expect-error on the undocumented parameter. This library doesn't validate at runtime that the request matches the type, so any extra values you send will be sent as-is.

client.chat.completions.create({
  // ...
  // @ts-expect-error baz is not yet public
  baz: 'undocumented option',
});
For requests with the GET verb, any extra params will be in the query, all other requests will send the extra param in the body.

If you want to explicitly send an extra argument, you can do so with the query, body, and headers request options.

Undocumented response properties
To access undocumented response properties, you may access the response object with // @ts-expect-error on the response object, or cast the response object to the requisite type. Like the request params, we do not validate or strip extra properties from the response from the API.

Customizing the fetch client
If you want to use a different fetch function, you can either polyfill the global:

import fetch from 'my-fetch';

globalThis.fetch = fetch;
Or pass it to the client:

import OpenAI from 'openai';
import fetch from 'my-fetch';

const client = new OpenAI({ fetch });
Fetch options
If you want to set custom fetch options without overriding the fetch function, you can provide a fetchOptions object when instantiating the client or making a request. (Request-specific options override client options.)

import OpenAI from 'openai';

const client = new OpenAI({
  fetchOptions: {
    // `RequestInit` options
  },
});
Configuring proxies
To modify proxy behavior, you can provide custom fetchOptions that add runtime-specific proxy options to requests:

 Node [docs]

import OpenAI from 'openai';
import * as undici from 'undici';

const proxyAgent = new undici.ProxyAgent('http://localhost:8888');
const client = new OpenAI({
  fetchOptions: {
    dispatcher: proxyAgent,
  },
});
 Bun [docs]

import OpenAI from 'openai';

const client = new OpenAI({
  fetchOptions: {
    proxy: 'http://localhost:8888',
  },
});
 Deno [docs]

import OpenAI from 'npm:openai';

const httpClient = Deno.createHttpClient({ proxy: { url: 'http://localhost:8888' } });
const client = new OpenAI({
  fetchOptions: {
    client: httpClient,
  },
});
Frequently Asked Questions
Semantic versioning
This package generally follows SemVer conventions, though certain backwards-incompatible changes may be released as minor versions:

Changes that only affect static types, without breaking runtime behavior.
Changes to library internals which are technically public but not intended or documented for external use. (Please open a GitHub issue to let us know if you are relying on such internals.)
Changes that we do not expect to impact the vast majority of users in practice.
We take backwards-compatibility seriously and work hard to ensure you can rely on a smooth upgrade experience.

We are keen for your feedback; please open an issue with questions, bugs, or suggestions.

Requirements
TypeScript >= 4.9 is supported.

The following runtimes are supported:

Node.js 20 LTS or later (non-EOL) versions.

Deno v1.28.0 or higher.

Bun 1.0 or later.

Cloudflare Workers.

Vercel Edge Runtime.

Jest 28 or greater with the "node" environment ("jsdom" is not supported at this time).

Nitro v2.6 or greater.

Web browsers: disabled by default to avoid exposing your secret API credentials. Enable browser support by explicitly setting dangerouslyAllowBrowser to true'.

More explanation
Note that React Native is not supported at this time.

If you are interested in other runtime environments, please open or upvote an issue on GitHub.

Contributing
See the contributing documentation.

Readme
Keywords
none
Package Sidebar
Install
npm i openai


Repository
github.com/openai/openai-node

Homepage
github.com/openai/openai-node#readme

Weekly Downloads
12,066,073

Version
6.17.0

License
Apache-2.0

Unpacked Size
7.16 MB

Total Files
1995

Last publish
8 days ago

Collaborators
tylersmith-openai
atty-openai
tibo-openai
dkundel-openai
mbolin-openai
fouad-openai
easong-openai
aibrahim-openai
apcha-oai
seratch-openai
gabor-openai
dschnurr
jeevnayak
knight-oai
dschnurr-openai
kwhinnery-openai
Try on RunKit
Report malwarepful assistant.",
    "input": "Hello!",
    "stream": true
  }'
Response
event: response.created
data: {"type":"response.created","response":{"id":"resp_67c9fdcecf488190bdd9a0409de3a1ec07b8b0ad4e5eb654","object":"response","created_at":1741290958,"status":"in_progress","error":null,"incomplete_details":null,"instructions":"You are a helpful assistant.","max_output_tokens":null,"model":"gpt-4.1-2025-04-14","output":[],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{"effort":null,"summary":null},"store":true,"temperature":1.0,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1.0,"truncation":"disabled","usage":null,"user":null,"metadata":{}}}

event: response.in_progress
data: {"type":"response.in_progress","response":{"id":"resp_67c9fdcecf488190bdd9a0409de3a1ec07b8b0ad4e5eb654","object":"response","created_at":1741290958,"status":"in_progress","error":null,"incomplete_details":null,"instructions":"You are a helpful assistant.","max_output_tokens":null,"model":"gpt-4.1-2025-04-14","output":[],"parallel_tool_calls":true,"previous_response_id":null,"reasoning":{"effort":null,"summary":null},"store":true,"temperature":1.0,"text":{"format":{"type":"text"}},"tool_choice":"auto","tools":[],"top_p":1.0,"truncation":"disabled","usage":null,"user":null,"metadata":{}}}

event: response.output_item.added
data: {"type":"response.output_item.added","output_index":0,"item":{"id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654","type":"message","status":"in_progress","role":"assistant","content":[]}}

event: response.content_part.added
data: {"type":"response.content_part.added","item_id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}

event: response.output_text.delta
data: {"type":"response.output_text.delta","item_id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654","output_index":0,"content_index":0,"delta":"Hi"}

...

event: response.output_text.done
data: {"type":"response.output_text.done","item_id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654","output_index":0,"content_index":0,"text":"Hi there! How can I assist you today?"}

event: response.content_part.done
data: {"type":"response.content_part.done","item_id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654","output_index":0,"content_index":0,"part":{"type":"output_text","text":"Hi there! How can I assist you today?","annotations":[]}}

event: response.output_item.done
data: {"type":"response.output_item.done","output_index":0,"item":{"id":"msg_67c9fdcf37fc8190ba82116e33fb28c507b8b0ad4e5eb654","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hi there! How can I assist you today?","annotations":[]}]}}

event: response.completed
data: {"type":"response.completed","response":