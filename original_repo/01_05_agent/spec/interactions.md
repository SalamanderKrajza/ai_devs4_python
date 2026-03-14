The Interactions API ([Beta](https://ai.google.dev/gemini-api/docs/api-versions)) is a unified interface
for interacting with Gemini models and agents. As an improved alternative
to the [`generateContent`](https://ai.google.dev/api/generate-content#method:-models.generatecontent)
API, it simplifies state management, tool orchestration,
and long-running tasks. For comprehensive view of the API schema, see the
[API Reference](https://ai.google.dev/api/interactions-api). During the Beta,
features and schemas are subject to [breaking changes](https://ai.google.dev/gemini-api/docs/interactions#breaking-changes).
To get started quickly, try the [Interactions API quickstart notebook](https://colab.sandbox.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Get_started_interactions_api.ipynb).

General use Function calling Deep Research agent

The following example shows how to call the Interactions API with a text prompt.  

### Python

    from google import genai

    client = genai.Client()

    interaction =  client.interactions.create(
        model="gemini-3-flash-preview",
        input="Tell me a short joke about programming."
    )

    print(interaction.outputs[-1].text)

### JavaScript

    import { GoogleGenAI } from '@google/genai';

    const client = new GoogleGenAI({});

    const interaction =  await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: 'Tell me a short joke about programming.',
    });

    console.log(interaction.outputs[interaction.outputs.length - 1].text);

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-3-flash-preview",
        "input": "Tell me a short joke about programming."
    }'

## Basic interactions

The Interactions API is available through our
[existing SDKs](https://ai.google.dev/gemini-api/docs/interactions#sdk). The simplest way to interact with the model is by
providing a text prompt. The `input` can be a string, a list containing a
content objects, or a list of turns with roles and content objects.  

### Python

    from google import genai

    client = genai.Client()

    interaction =  client.interactions.create(
        model="gemini-3-flash-preview",
        input="Tell me a short joke about programming."
    )

    print(interaction.outputs[-1].text)

### JavaScript

    import { GoogleGenAI } from '@google/genai';

    const client = new GoogleGenAI({});

    const interaction =  await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: 'Tell me a short joke about programming.',
    });

    console.log(interaction.outputs[interaction.outputs.length - 1].text);

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-3-flash-preview",
        "input": "Tell me a short joke about programming."
    }'

| **Note:** Interaction objects are saved by default (`store=true`) to enable state management features and background execution. See [Data Storage and
| Retention](https://ai.google.dev/gemini-api/docs/interactions#data-storage-retention) for details on retention periods and how to delete stored data or opt out.

## Conversation

You can build multi-turn conversations in two ways:

- Statefully by referencing a previous interaction
- Statelessly by providing the entire conversation history

### Stateful conversation

Pass the `id` from the previous interaction to the `previous_interaction_id`
parameter to continue a conversation.  

### Python

    from google import genai

    client = genai.Client()

    # 1. First turn
    interaction1 = client.interactions.create(
        model="gemini-3-flash-preview",
        input="Hi, my name is Phil."
    )
    print(f"Model: {interaction1.outputs[-1].text}")

    # 2. Second turn (passing previous_interaction_id)
    interaction2 = client.interactions.create(
        model="gemini-3-flash-preview",
        input="What is my name?",
        previous_interaction_id=interaction1.id
    )
    print(f"Model: {interaction2.outputs[-1].text}")

### JavaScript

    import { GoogleGenAI } from '@google/genai';

    const client = new GoogleGenAI({});

    // 1. First turn
    const interaction1 = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: 'Hi, my name is Phil.'
    });
    console.log(`Model: ${interaction1.outputs[interaction1.outputs.length - 1].text}`);

    // 2. Second turn (passing previous_interaction_id)
    const interaction2 = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: 'What is my name?',
        previous_interaction_id: interaction1.id
    });
    console.log(`Model: ${interaction2.outputs[interaction2.outputs.length - 1].text}`);

### REST

    # 1. First turn
    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-3-flash-preview",
        "input": "Hi, my name is Phil."
    }'

    # 2. Second turn (Replace INTERACTION_ID with the ID from the previous interaction)
    # curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    # -H "Content-Type: application/json" \
    # -H "x-goog-api-key: $GEMINI_API_KEY" \
    # -d '{
    #     "model": "gemini-3-flash-preview",
    #     "input": "What is my name?",
    #     "previous_interaction_id": "INTERACTION_ID"
    # }'

#### Retrieve past stateful interactions

Using the interaction `id` to retrieve previous turns of the conversation.  

### Python

    previous_interaction = client.interactions.get("<YOUR_INTERACTION_ID>")

    print(previous_interaction)

### JavaScript

    const previous_interaction = await client.interactions.get("<YOUR_INTERACTION_ID>");
    console.log(previous_interaction);

### REST

    curl -X GET "https://generativelanguage.googleapis.com/v1beta/interactions/<YOUR_INTERACTION_ID>" \
    -H "x-goog-api-key: $GEMINI_API_KEY"

### Stateless conversation

You can manage conversation history manually on the client side.  

### Python

    from google import genai

    client = genai.Client()

    conversation_history = [
        {
            "role": "user",
            "content": "What are the three largest cities in Spain?"
        }
    ]

    interaction1 = client.interactions.create(
        model="gemini-3-flash-preview",
        input=conversation_history
    )

    print(f"Model: {interaction1.outputs[-1].text}")

    conversation_history.append({"role": "model", "content": interaction1.outputs})
    conversation_history.append({
        "role": "user",
        "content": "What is the most famous landmark in the second one?"
    })

    interaction2 = client.interactions.create(
        model="gemini-3-flash-preview",
        input=conversation_history
    )

    print(f"Model: {interaction2.outputs[-1].text}")

### JavaScript

    import { GoogleGenAI } from '@google/genai';

    const client = new GoogleGenAI({});

    const conversationHistory = [
        {
            role: 'user',
            content: "What are the three largest cities in Spain?"
        }
    ];

    const interaction1 = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: conversationHistory
    });

    console.log(`Model: ${interaction1.outputs[interaction1.outputs.length - 1].text}`);

    conversationHistory.push({ role: 'model', content: interaction1.outputs });
    conversationHistory.push({
        role: 'user',
        content: "What is the most famous landmark in the second one?"
    });

    const interaction2 = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: conversationHistory
    });

    console.log(`Model: ${interaction2.outputs[interaction2.outputs.length - 1].text}`);

### REST

     curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
     -H "Content-Type: application/json" \
     -H "x-goog-api-key: $GEMINI_API_KEY" \
     -d '{
        "model": "gemini-3-flash-preview",
        "input": [
            {
                "role": "user",
                "content": "What are the three largest cities in Spain?"
            },
            {
                "role": "model",
                "content": "The three largest cities in Spain are Madrid, Barcelona, and Valencia."
            },
            {
                "role": "user",
                "content": "What is the most famous landmark in the second one?"
            }
        ]
    }'

## Multimodal capabilities

You can use the Interactions API for multimodal use cases such as image
understanding or video generation.

### Multimodal understanding

You can provide multimodal input as base64-encoded data inline, by using the
Files API for larger files, or by passing a publicly accessible link in the uri
field. The code samples that follow demonstrate the public URL method.

#### Image understanding

### Python

    from google import genai
    client = genai.Client()

    interaction = client.interactions.create(
        model="gemini-3-flash-preview",
        input=[
            {"type": "text", "text": "Describe the image."},
            {
                "type": "image",
                "uri": "<var translate="no">YOUR_URL</var>",
                "mime_type": "image/png"
            }
        ]
    )
    print(interaction.outputs[-1].text)

### JavaScript

    import {GoogleGenAI} from '@google/genai';

    const client = new GoogleGenAI({});

    const interaction = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: [
            {type: 'text', text: 'Describe the image.'},
            {
                type: 'image',
                uri: '<var translate="no">YOUR_URL</var>',
                mime_type: 'image/png'
            }
        ]
    });
    console.log(interaction.outputs[interaction.outputs.length - 1].text);

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "gemini-3-flash-preview",
        "input": [
        {
            "type": "text",
            "text": "Describe the image."
        },
        {
            "type": "image",
            "uri": "<var translate="no">YOUR_URL</var>",
            "mime_type": "image/png"
        }
        ]
    }'

#### Audio understanding

### Python

    from google import genai
    client = genai.Client()

    interaction = client.interactions.create(
        model="gemini-3-flash-preview",
        input=[
            {"type": "text", "text": "What does this audio say?"},
            {
                "type": "audio",
                "uri": "<var translate="no">YOUR_URL</var>",
                "mime_type": "audio/wav"
            }
        ]
    )
    print(interaction.outputs[-1].text)

### JavaScript

    import { GoogleGenAI } from '@google/genai';

    const client = new GoogleGenAI({});

    const interaction = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: [
            { type: 'text', text: 'What does this audio say?' },
            {
                type: 'audio',
                uri: '<var translate="no">YOUR_URL</var>',
                mime_type: 'audio/wav'
            }
        ]
    });

    console.log(interaction.outputs[interaction.outputs.length - 1].text);

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-3-flash-preview",
        "input": [
            {"type": "text", "text": "What does this audio say?"},
            {
                "type": "audio",
                "uri": "<var translate="no">YOUR_URL</var>",
                "mime_type": "audio/wav"
            }
        ]
    }'

#### Video understanding

### Python

    from google import genai
    client = genai.Client()

    print("Analyzing video...")
    interaction = client.interactions.create(
        model="gemini-3-flash-preview",
        input=[
            {"type": "text", "text": "What is happening in this video? Provide a timestamped summary."},
            {
                "type": "video",
                "uri": "<var translate="no">YOUR_URL</var>",
                "mime_type": "video/mp4"
            }
        ]
    )

    print(interaction.outputs[-1].text)

### JavaScript

    import { GoogleGenAI } from '@google/genai';

    const client = new GoogleGenAI({});

    console.log('Analyzing video...');
    const interaction = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: [
            { type: 'text', text: 'What is happening in this video? Provide a timestamped summary.' },
            {
                type: 'video',
                uri: '<var translate="no">YOUR_URL</var>',
                mime_type: 'video/mp4'
            }
        ]
    });

    console.log(interaction.outputs[interaction.outputs.length - 1].text);

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-3-flash-preview",
        "input": [
            {"type": "text", "text": "What is happening in this video?"},
            {
                "type": "video",
                "uri": "<var translate="no">YOUR_URL</var>",
                "mime_type": "video/mp4"
            }
        ]
    }'

#### Document (PDF) understanding

### Python

    from google import genai
    client = genai.Client()

    interaction = client.interactions.create(
        model="gemini-3-flash-preview",
        input=[
            {"type": "text", "text": "What is this document about?"},
            {
                "type": "document",
                "uri": "<var translate="no">YOUR_URL</var>",
                "mime_type": "application/pdf"
            }
        ]
    )
    print(interaction.outputs[-1].text)

### JavaScript

    import { GoogleGenAI } from '@google/genai';

    const client = new GoogleGenAI({});

    const interaction = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: [
            { type: 'text', text: 'What is this document about?' },
            {
                type: 'document',
                uri: '<var translate="no">YOUR_URL</var>',
                mime_type: 'application/pdf'
            }
        ],
    });
    console.log(interaction.outputs[0].text);

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-3-flash-preview",
        "input": [
            {"type": "text", "text": "What is this document about?"},
            {
                "type": "document",
                "uri": "<var translate="no">YOUR_URL</var>",
                "mime_type": "application/pdf"
            }
        ]
    }'

### Multimodal generation

You can use Interactions API to generate multimodal outputs.

#### Image generation

### Python

    import base64
    from google import genai

    client = genai.Client()

    interaction = client.interactions.create(
        model="gemini-3-pro-image-preview",
        input="Generate an image of a futuristic city.",
        response_modalities=["IMAGE"]
    )

    for output in interaction.outputs:
        if output.type == "image":
            print(f"Generated image with mime_type: {output.mime_type}")
            # Save the image
            with open("generated_city.png", "wb") as f:
                f.write(base64.b64decode(output.data))

### JavaScript

    import { GoogleGenAI } from '@google/genai';
    import * as fs from 'fs';

    const client = new GoogleGenAI({});

    const interaction = await client.interactions.create({
        model: 'gemini-3-pro-image-preview',
        input: 'Generate an image of a futuristic city.',
        response_modalities: ['IMAGE']
    });

    for (const output of interaction.outputs) {
        if (output.type === 'image') {
            console.log(`Generated image with mime_type: ${output.mime_type}`);
            // Save the image
            fs.writeFileSync('generated_city.png', Buffer.from(output.data, 'base64'));
        }
    }

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-3-pro-image-preview",
        "input": "Generate an image of a futuristic city.",
        "response_modalities": ["IMAGE"]
    }'

##### Configure image output

You can customize generated images using `image_config` within `generation_config`
to control the aspect ratio and resolution.

| Parameter | Options | Description |
|---|---|---|
| `aspect_ratio` | `1:1`, `2:3`, `3:2`, `3:4`, `4:3`, `4:5`, `5:4`, `9:16`, `16:9`, `21:9` | Controls the width-to-height ratio of the output image. |
| `image_size` | `1k`, `2k`, `4k` | Sets the output image resolution. |

### Python

    import base64
    from google import genai

    client = genai.Client()

    interaction = client.interactions.create(
        model="gemini-3-pro-image-preview",
        input="Generate an image of a futuristic city.",
        generation_config={
            "image_config": {
                "aspect_ratio": "9:16",
                "image_size": "2k"
            }
        }
    )

    for output in interaction.outputs:
        if output.type == "image":
            print(f"Generated image with mime_type: {output.mime_type}")
            # Save the image
            with open("generated_city.png", "wb") as f:
                f.write(base64.b64decode(output.data))

### JavaScript

    import { GoogleGenAI } from '@google/genai';
    import * as fs from 'fs';

    const client = new GoogleGenAI({});

    const interaction = await client.interactions.create({
        model: 'gemini-3-pro-image-preview',
        input: 'Generate an image of a futuristic city.',
        generation_config: {
            image_config: {
                aspect_ratio: '9:16',
                image_size: '2k'
            }
        }
    });

    for (const output of interaction.outputs) {
        if (output.type === 'image') {
            console.log(`Generated image with mime_type: ${output.mime_type}`);
            // Save the image
            fs.writeFileSync('generated_city.png', Buffer.from(output.data, 'base64'));
        }
    }

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-3-pro-image-preview",
        "input": "Generate an image of a futuristic city.",
        "generation_config": {
            "image_config": {
                "aspect_ratio": "9:16",
                "image_size": "2k"
            }
        }
    }'

#### Speech generation

Generate natural-sounding speech from text using the text-to-speech (TTS) model.
Configure voice, language, and speaker settings with the `speech_config`   

parameter.  

### Python

    import base64
    from google import genai
    import wave

    # Set up the wave file to save the output:
    def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm)

    client = genai.Client()

    interaction = client.interactions.create(
        model="gemini-2.5-flash-preview-tts",
        input="Say the following: WOOHOO This is so much fun!.",
        response_modalities=["AUDIO"],
        generation_config={
            "speech_config": {
                "language": "en-us",
                "voice": "kore"
            }
        }
    )

    for output in interaction.outputs:
        if output.type == "audio":
            print(f"Generated audio with mime_type: {output.mime_type}")
            # Save the audio as wave file to the current directory.
            wave_file("generated_audio.wav", base64.b64decode(output.data))

### JavaScript

    import { GoogleGenAI } from '@google/genai';
    import * as fs from 'fs';
    import wav from 'wav';

    async function saveWaveFile(
        filename,
        pcmData,
        channels = 1,
        rate = 24000,
        sampleWidth = 2,
    ) {
        return new Promise((resolve, reject) => {
            const writer = new wav.FileWriter(filename, {
                    channels,
                    sampleRate: rate,
                    bitDepth: sampleWidth * 8,
            });

            writer.on('finish', resolve);
            writer.on('error', reject);

            writer.write(pcmData);
            writer.end();
        });
    }

    async function main() {
        const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
        const client = new GoogleGenAI({apiKey: GEMINI_API_KEY});

        const interaction = await client.interactions.create({
            model: 'gemini-2.5-flash-preview-tts',
            input: 'Say the following: WOOHOO This is so much fun!.',
            response_modalities: ['AUDIO'],
            generation_config: {
                speech_config: {
                    language: "en-us",
                    voice: "kore"
                }
            }
        });

        for (const output of interaction.outputs) {
            if (output.type === 'audio') {
                console.log(`Generated audio with mime_type: ${output.mime_type}`);
                const audioBuffer = Buffer.from(output.data, 'base64');
                // Save the audio as wave file to the current directory
                await saveWaveFile("generated_audio.wav", audioBuffer);
            }
        }
    }
    await main();

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-2.5-flash-preview-tts",
        "input": "Say the following: WOOHOO This is so much fun!.",
        "response_modalities": ["AUDIO"],
        "generation_config": {
            "speech_config": {
                "language": "en-us",
                "voice": "kore"
            }
        }
    }' | jq -r '.outputs[] | select(.type == "audio") | .data' | base64 -d > generated_audio.pcm
    # You may need to install ffmpeg.
    ffmpeg -f s16le -ar 24000 -ac 1 -i generated_audio.pcm generated_audio.wav

##### Multi-speaker speech generation

Generate speech with multiple speakers by specifying speaker names in the prompt
and matching them in the `speech_config`.

The prompt should include the speaker names:  

    TTS the following conversation between Alice and Bob:
    Alice: Hi Bob, how are you doing today?
    Bob: I'm doing great, thanks for asking! How about you?
    Alice: Fantastic! I just learned about the Gemini API.

Then configure the `speech_config` with matching speakers:  

    "generation_config": {
        "speech_config": [
            {"voice": "Zephyr", "speaker": "Alice", "language": "en-US"},
            {"voice": "Puck", "speaker": "Bob", "language": "en-US"}
        ]
    }

## Agentic capabilities

The Interactions API is designed for building and interacting with agents, and
includes support for function calling, built-in tools, structured outputs, and
the Model Context Protocol (MCP).

### Agents

You can use specialized agents like `deep-research-pro-preview-12-2025` for
complex tasks. To learn more about the Gemini Deep Research Agent, see the
[Deep Research](https://ai.google.dev/gemini-api/docs/deep-research) guide.
**Note:** The `background=true` parameter is only supported for agents.  

### Python

    import time
    from google import genai

    client = genai.Client()

    # 1. Start the Deep Research Agent
    initial_interaction = client.interactions.create(
        input="Research the history of the Google TPUs with a focus on 2025 and 2026.",
        agent="deep-research-pro-preview-12-2025",
        background=True
    )

    print(f"Research started. Interaction ID: {initial_interaction.id}")

    # 2. Poll for results
    while True:
        interaction = client.interactions.get(initial_interaction.id)
        print(f"Status: {interaction.status}")

        if interaction.status == "completed":
            print("\nFinal Report:\n", interaction.outputs[-1].text)
            break
        elif interaction.status in ["failed", "cancelled"]:
            print(f"Failed with status: {interaction.status}")
            break

        time.sleep(10)

### JavaScript

    import { GoogleGenAI } from '@google/genai';

    const client = new GoogleGenAI({});

    // 1. Start the Deep Research Agent
    const initialInteraction = await client.interactions.create({
        input: 'Research the history of the Google TPUs with a focus on 2025 and 2026.',
        agent: 'deep-research-pro-preview-12-2025',
        background: true
    });

    console.log(`Research started. Interaction ID: ${initialInteraction.id}`);

    // 2. Poll for results
    while (true) {
        const interaction = await client.interactions.get(initialInteraction.id);
        console.log(`Status: ${interaction.status}`);

        if (interaction.status === 'completed') {
            console.log('\nFinal Report:\n', interaction.outputs[interaction.outputs.length - 1].text);
            break;
        } else if (['failed', 'cancelled'].includes(interaction.status)) {
            console.log(`Failed with status: ${interaction.status}`);
            break;
        }

        await new Promise(resolve => setTimeout(resolve, 10000));
    }

### REST

    # 1. Start the Deep Research Agent
    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "input": "Research the history of the Google TPUs with a focus on 2025 and 2026.",
        "agent": "deep-research-pro-preview-12-2025",
        "background": true
    }'

    # 2. Poll for results (Replace INTERACTION_ID with the ID from the previous interaction)
    # curl -X GET "https://generativelanguage.googleapis.com/v1beta/interactions/INTERACTION_ID" \
    # -H "x-goog-api-key: $GEMINI_API_KEY"

### Tools and function calling

This section explains how to use function calling to define custom tools and how
to use Google's built-in tools within the Interactions API.

#### Function calling

### Python

    from google import genai

    client = genai.Client()

    # 1. Define the tool
    def get_weather(location: str):
        """Gets the weather for a given location."""
        return f"The weather in {location} is sunny."

    weather_tool = {
        "type": "function",
        "name": "get_weather",
        "description": "Gets the weather for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}
            },
            "required": ["location"]
        }
    }

    # 2. Send the request with tools
    interaction = client.interactions.create(
        model="gemini-3-flash-preview",
        input="What is the weather in Paris?",
        tools=[weather_tool]
    )

    # 3. Handle the tool call
    for output in interaction.outputs:
        if output.type == "function_call":
            print(f"Tool Call: {output.name}({output.arguments})")
            # Execute tool
            result = get_weather(**output.arguments)

            # Send result back
            interaction = client.interactions.create(
                model="gemini-3-flash-preview",
                previous_interaction_id=interaction.id,
                input=[{
                    "type": "function_result",
                    "name": output.name,
                    "call_id": output.id,
                    "result": result
                }]
            )
            print(f"Response: {interaction.outputs[-1].text}")

### JavaScript

    import { GoogleGenAI } from '@google/genai';

    const client = new GoogleGenAI({});

    // 1. Define the tool
    const weatherTool = {
        type: 'function',
        name: 'get_weather',
        description: 'Gets the weather for a given location.',
        parameters: {
            type: 'object',
            properties: {
                location: { type: 'string', description: 'The city and state, e.g. San Francisco, CA' }
            },
            required: ['location']
        }
    };

    // 2. Send the request with tools
    let interaction = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: 'What is the weather in Paris?',
        tools: [weatherTool]
    });

    // 3. Handle the tool call
    for (const output of interaction.outputs) {
        if (output.type === 'function_call') {
            console.log(`Tool Call: ${output.name}(${JSON.stringify(output.arguments)})`);

            // Execute tool (Mocked)
            const result = `The weather in ${output.arguments.location} is sunny.`;

            // Send result back
            interaction = await client.interactions.create({
                model: 'gemini-3-flash-preview',
                previous_interaction_id:interaction.id,
                input: [{
                    type: 'function_result',
                    name: output.name,
                    call_id: output.id,
                    result: result
                }]
            });
            console.log(`Response: ${interaction.outputs[interaction.outputs.length - 1].text}`);
        }
    }

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-3-flash-preview",
        "input": "What is the weather in Paris?",
        "tools": [{
            "type": "function",
            "name": "get_weather",
            "description": "Gets the weather for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}
                },
                "required": ["location"]
            }
        }]
    }'

    # Handle the tool call and send result back (Replace INTERACTION_ID and CALL_ID)
    # curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    # -H "Content-Type: application/json" \
    # -H "x-goog-api-key: $GEMINI_API_KEY" \
    # -d '{
    #     "model": "gemini-3-flash-preview",
    #     "previous_interaction_id": "INTERACTION_ID",
    #     "input": [{
    #         "type": "function_result",
    #         "name": "get_weather",
    #         "call_id": "FUNCTION_CALL_ID",
    #         "result": "The weather in Paris is sunny."
    #     }]
    # }'

##### Function calling with client-side state

If you don't want to use server-side state, you can manage it all
on the client side.  

### Python

    from google import genai
    client = genai.Client()

    functions = [
        {
            "type": "function",
            "name": "schedule_meeting",
            "description": "Schedules a meeting with specified attendees at a given time and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "attendees": {"type": "array", "items": {"type": "string"}},
                    "date": {"type": "string", "description": "Date of the meeting (e.g., 2024-07-29)"},
                    "time": {"type": "string", "description": "Time of the meeting (e.g., 15:00)"},
                    "topic": {"type": "string", "description": "The subject of the meeting."},
                },
                "required": ["attendees", "date", "time", "topic"],
            },
        }
    ]

    history = [{"role": "user","content": [{"type": "text", "text": "Schedule a meeting for 2025-11-01 at 10 am with Peter and Amir about the Next Gen API."}]}]

    # 1. Model decides to call the function
    interaction = client.interactions.create(
        model="gemini-3-flash-preview",
        input=history,
        tools=functions
    )

    # add model interaction back to history
    history.append({"role": "model", "content": interaction.outputs})

    for output in interaction.outputs:
        if output.type == "function_call":
            print(f"Function call: {output.name} with arguments {output.arguments}")

            # 2. Execute the function and get a result
            # In a real app, you would call your function here.
            # call_result = schedule_meeting(**json.loads(output.arguments))
            call_result = "Meeting scheduled successfully."

            # 3. Send the result back to the model
            history.append({"role": "user", "content": [{"type": "function_result", "name": output.name, "call_id": output.id, "result": call_result}]})

            interaction2 = client.interactions.create(
                model="gemini-3-flash-preview",
                input=history,
            )
            print(f"Final response: {interaction2.outputs[-1].text}")
        else:
            print(f"Output: {output}")

### JavaScript

    // 1. Define the tool
    const functions = [
        {
            type: 'function',
            name: 'schedule_meeting',
            description: 'Schedules a meeting with specified attendees at a given time and date.',
            parameters: {
                type: 'object',
                properties: {
                    attendees: { type: 'array', items: { type: 'string' } },
                    date: { type: 'string', description: 'Date of the meeting (e.g., 2024-07-29)' },
                    time: { type: 'string', description: 'Time of the meeting (e.g., 15:00)' },
                    topic: { type: 'string', description: 'The subject of the meeting.' },
                },
                required: ['attendees', 'date', 'time', 'topic'],
            },
        },
    ];

    const history = [
        { role: 'user', content: [{ type: 'text', text: 'Schedule a meeting for 2025-11-01 at 10 am with Peter and Amir about the Next Gen API.' }] }
    ];

    // 2. Model decides to call the function
    let interaction = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: history,
        tools: functions
    });

    // add model interaction back to history
    history.push({ role: 'model', content: interaction.outputs });

    for (const output of interaction.outputs) {
        if (output.type === 'function_call') {
            console.log(`Function call: ${output.name} with arguments ${JSON.stringify(output.arguments)}`);

            // 3. Send the result back to the model
            history.push({ role: 'user', content: [{ type: 'function_result', name: output.name, call_id: output.id, result: 'Meeting scheduled successfully.' }] });

            const interaction2 = await client.interactions.create({
                model: 'gemini-3-flash-preview',
                input: history,
            });
            console.log(`Final response: ${interaction2.outputs[interaction2.outputs.length - 1].text}`);
        }
    }

#### Built-in tools

Gemini comes with built-in tools like
[Grounding with Google Search](https://ai.google.dev/gemini-api/docs/google-search),
[Code execution](https://ai.google.dev/gemini-api/docs/code-execution),
[URL context](https://ai.google.dev/gemini-api/docs/url-context), and
[Computer Use](https://ai.google.dev/gemini-api/docs/computer-use)

##### Grounding with Google Search

### Python

    from google import genai

    client = genai.Client()

    interaction = client.interactions.create(
        model="gemini-3-flash-preview",
        input="Who won the last Super Bowl?",
        tools=[{"type": "google_search"}]
    )
    # Find the text output (not the GoogleSearchResultContent)
    text_output = next((o for o in interaction.outputs if o.type == "text"), None)
    if text_output:
        print(text_output.text)

### JavaScript

    import { GoogleGenAI } from '@google/genai';

    const client = new GoogleGenAI({});

    const interaction = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: 'Who won the last Super Bowl?',
        tools: [{ type: 'google_search' }]
    });
    // Find the text output (not the GoogleSearchResultContent)
    const textOutput = interaction.outputs.find(o => o.type === 'text');
    if (textOutput) console.log(textOutput.text);

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-3-flash-preview",
        "input": "Who won the last Super Bowl?",
        "tools": [{"type": "google_search"}]
    }'

##### Code execution

### Python

    from google import genai

    client = genai.Client()

    interaction = client.interactions.create(
        model="gemini-3-flash-preview",
        input="Calculate the 50th Fibonacci number.",
        tools=[{"type": "code_execution"}]
    )
    print(interaction.outputs[-1].text)

### JavaScript

    import { GoogleGenAI } from '@google/genai';

    const client = new GoogleGenAI({});

    const interaction = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: 'Calculate the 50th Fibonacci number.',
        tools: [{ type: 'code_execution' }]
    });
    console.log(interaction.outputs[interaction.outputs.length - 1].text);

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-3-flash-preview",
        "input": "Calculate the 50th Fibonacci number.",
        "tools": [{"type": "code_execution"}]
    }'

##### URL context

### Python

    from google import genai

    client = genai.Client()

    interaction = client.interactions.create(
        model="gemini-3-flash-preview",
        input="Summarize the content of https://www.wikipedia.org/",
        tools=[{"type": "url_context"}]
    )
    # Find the text output (not the URLContextResultContent)
    text_output = next((o for o in interaction.outputs if o.type == "text"), None)
    if text_output:
        print(text_output.text)

### JavaScript

    import { GoogleGenAI } from '@google/genai';

    const client = new GoogleGenAI({});

    const interaction = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: 'Summarize the content of https://www.wikipedia.org/',
        tools: [{ type: 'url_context' }]
    });
    // Find the text output (not the URLContextResultContent)
    const textOutput = interaction.outputs.find(o => o.type === 'text');
    if (textOutput) console.log(textOutput.text);

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-3-flash-preview",
        "input": "Summarize the content of https://www.wikipedia.org/",
        "tools": [{"type": "url_context"}]
    }'

##### Computer use

### Python

    from google import genai

    client = genai.Client()

    interaction = client.interactions.create(
        model="gemini-2.5-computer-use-preview-10-2025",
        input="Search for highly rated smart fridges with touchscreen, 2 doors, around 25 cu ft, priced below 4000 dollars on Google Shopping. Create a bulleted list of the 3 cheapest options in the format of name, description, price in an easy-to-read layout.",
        tools=[{
            "type": "computer_use",
            "environment": "browser",
            "excludedPredefinedFunctions": ["drag_and_drop"]
        }]
    )

    # The response will contain tool calls (actions) for the computer interface
    # or text explaining the action
    for output in interaction.outputs:
        print(output)

### JavaScript

    import { GoogleGenAI } from '@google/genai';

    const client = new GoogleGenAI({});

    const interaction = await client.interactions.create({
        model: 'gemini-2.5-computer-use-preview-10-2025',
        input: 'Search for highly rated smart fridges with touchscreen, 2 doors, around 25 cu ft, priced below 4000 dollars on Google Shopping. Create a bulleted list of the 3 cheapest options in the format of name, description, price in an easy-to-read layout.',
        tools: [{
            type: 'computer_use',
            environment: 'browser',
            excludedPredefinedFunctions: ['drag_and_drop']
        }]
    });

    // The response will contain tool calls (actions) for the computer interface
    // or text explaining the action
    interaction.outputs.forEach(output => console.log(output));

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-2.5-computer-use-preview-10-2025",
        "input": "Search for highly rated smart fridges with touchscreen, 2 doors, around 25 cu ft, priced below 4000 dollars on Google Shopping. Create a bulleted list of the 3 cheapest options in the format of name, description, price in an easy-to-read layout.",
        "tools": [{
            "type": "computer_use",
            "environment": "browser",
            "excludedPredefinedFunctions": ["drag_and_drop"]
        }]
    }'

### Remote Model context protocol (MCP)

Remote [MCP](https://modelcontextprotocol.io/docs/getting-started/intro)
integration simplifies agent development by allowing the Gemini API
to directly call external tools hosted on remote servers.  

### Python

    import datetime
    from google import genai

    client = genai.Client()

    mcp_server = {
        "type": "mcp_server",
        "name": "weather_service",
        "url": "https://gemini-api-demos.uc.r.appspot.com/mcp"
    }

    today = datetime.date.today().strftime("%d %B %Y")

    interaction = client.interactions.create(
        model="gemini-2.5-flash",
        input="What is the weather like in New York today?",
        tools=[mcp_server],
        system_instruction=f"Today is {today}."
    )

    print(interaction.outputs[-1].text)

### JavaScript

    import { GoogleGenAI } from '@google/genai';

    const client = new GoogleGenAI({});

    const mcpServer = {
        type: 'mcp_server',
        name: 'weather_service',
        url: 'https://gemini-api-demos.uc.r.appspot.com/mcp'
    };

    const today = new Date().toDateString();

    const interaction = await client.interactions.create({
        model: 'gemini-2.5-flash',
        input: 'What is the weather like in New York today?',
        tools: [mcpServer],
        system_instruction: `Today is ${today}.`
    });

    console.log(interaction.outputs[interaction.outputs.length - 1].text);

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-2.5-flash",
        "input": "What is the weather like in New York today?",
        "tools": [{
            "type": "mcp_server",
            "name": "weather_service",
            "url": "https://gemini-api-demos.uc.r.appspot.com/mcp"
        }],
        "system_instruction": "Today is '"$(date +"%du%Bt%Y")"' YYYY-MM-DD>."
    }'

**Important notes:**

- Remote MCP only works with Streamable HTTP servers (SSE servers are not supported)
- Remote MCP does not work with Gemini 3 models (this is coming soon)
- MCP server names shouldn't include "-" character (use snake_case server names instead)

### Structured output (JSON schema)

Enforce a specific JSON output by providing a JSON schema in the
`response_format` parameter. This is useful for tasks like moderation,
classification, or data extraction.  

### Python

    from google import genai
    from pydantic import BaseModel, Field
    from typing import Literal, Union
    client = genai.Client()

    class SpamDetails(BaseModel):
        reason: str = Field(description="The reason why the content is considered spam.")
        spam_type: Literal["phishing", "scam", "unsolicited promotion", "other"]

    class NotSpamDetails(BaseModel):
        summary: str = Field(description="A brief summary of the content.")
        is_safe: bool = Field(description="Whether the content is safe for all audiences.")

    class ModerationResult(BaseModel):
        decision: Union[SpamDetails, NotSpamDetails]

    interaction = client.interactions.create(
        model="gemini-3-flash-preview",
        input="Moderate the following content: 'Congratulations! You've won a free cruise. Click here to claim your prize: www.definitely-not-a-scam.com'",
        response_format=ModerationResult.model_json_schema(),
    )

    parsed_output = ModerationResult.model_validate_json(interaction.outputs[-1].text)
    print(parsed_output)

### JavaScript

    import { GoogleGenAI } from '@google/genai';
    import { z } from 'zod';
    const client = new GoogleGenAI({});

    const moderationSchema = z.object({
        decision: z.union([
            z.object({
                reason: z.string().describe('The reason why the content is considered spam.'),
                spam_type: z.enum(['phishing', 'scam', 'unsolicited promotion', 'other']).describe('The type of spam.'),
            }).describe('Details for content classified as spam.'),
            z.object({
                summary: z.string().describe('A brief summary of the content.'),
                is_safe: z.boolean().describe('Whether the content is safe for all audiences.'),
            }).describe('Details for content classified as not spam.'),
        ]),
    });

    const interaction = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: "Moderate the following content: 'Congratulations! You've won a free cruise. Click here to claim your prize: www.definitely-not-a-scam.com'",
        response_format: z.toJSONSchema(moderationSchema),
    });
    console.log(interaction.outputs[0].text);

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-3-flash-preview",
        "input": "Moderate the following content: 'Congratulations! You've won a free cruise. Click here to claim your prize: www.definitely-not-a-scam.com'",
        "response_format": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "The reason why the content is considered spam."},
                        "spam_type": {"type": "string", "description": "The type of spam."}
                    },
                    "required": ["reason", "spam_type"]
                }
            },
            "required": ["decision"]
        }
    }'

### Combining tools and structured output

Combine built-in tools with structured output to get a reliable JSON object
based on information retrieved by a tool.  

### Python

    from google import genai
    from pydantic import BaseModel, Field
    from typing import Literal, Union

    client = genai.Client()

    class SpamDetails(BaseModel):
        reason: str = Field(description="The reason why the content is considered spam.")
        spam_type: Literal["phishing", "scam", "unsolicited promotion", "other"]

    class NotSpamDetails(BaseModel):
        summary: str = Field(description="A brief summary of the content.")
        is_safe: bool = Field(description="Whether the content is safe for all audiences.")

    class ModerationResult(BaseModel):
        decision: Union[SpamDetails, NotSpamDetails]

    interaction = client.interactions.create(
        model="gemini-3-flash-preview",
        input="Moderate the following content: 'Congratulations! You've won a free cruise. Click here to claim your prize: www.definitely-not-a-scam.com'",
        response_format=ModerationResult.model_json_schema(),
        tools=[{"type": "url_context"}]
    )

    parsed_output = ModerationResult.model_validate_json(interaction.outputs[-1].text)
    print(parsed_output)

### JavaScript

    import { GoogleGenAI } from '@google/genai';
    import { z } from 'zod'; // Assuming zod is used for schema generation, or define manually
    const client = new GoogleGenAI({});

    const obj = z.object({
        winning_team: z.string(),
        score: z.string(),
    });
    const schema = z.toJSONSchema(obj);

    const interaction = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: 'Who won the last euro?',
        tools: [{ type: 'google_search' }],
        response_format: schema,
    });
    console.log(interaction.outputs[0].text);

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-3-flash-preview",
        "input": "Who won the last euro?",
        "tools": [{"type": "google_search"}],
        "response_format": {
            "type": "object",
            "properties": {
                "winning_team": {"type": "string"},
                "score": {"type": "string"}
            }
        }
    }'

## Advanced features

There are also additional advance features that give you more flexibility
in working with Interactions API.

### Streaming

Receive responses incrementally as they are generated.  

### Python

    from google import genai

    client = genai.Client()

    stream = client.interactions.create(
        model="gemini-3-flash-preview",
        input="Explain quantum entanglement in simple terms.",
        stream=True
    )

    for chunk in stream:
        if chunk.event_type == "content.delta":
            if chunk.delta.type == "text":
                print(chunk.delta.text, end="", flush=True)
            elif chunk.delta.type == "thought":
                print(chunk.delta.thought, end="", flush=True)
        elif chunk.event_type == "interaction.complete":
            print(f"\n\n--- Stream Finished ---")
            print(f"Total Tokens: {chunk.interaction.usage.total_tokens}")

### JavaScript

    import { GoogleGenAI } from '@google/genai';

    const client = new GoogleGenAI({});

    const stream = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: 'Explain quantum entanglement in simple terms.',
        stream: true,
    });

    for await (const chunk of stream) {
        if (chunk.event_type === 'content.delta') {
            if (chunk.delta.type === 'text' && 'text' in chunk.delta) {
                process.stdout.write(chunk.delta.text);
            } else if (chunk.delta.type === 'thought' && 'thought' in chunk.delta) {
                process.stdout.write(chunk.delta.thought);
            }
        } else if (chunk.event_type === 'interaction.complete') {
            console.log('\n\n--- Stream Finished ---');
            console.log(`Total Tokens: ${chunk.interaction.usage.total_tokens}`);
        }
    }

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions?alt=sse" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-3-flash-preview",
        "input": "Explain quantum entanglement in simple terms.",
        "stream": true
    }'

### Configuration

Customize the model's behavior with `generation_config`.  

### Python

    from google import genai

    client = genai.Client()

    interaction = client.interactions.create(
        model="gemini-3-flash-preview",
        input="Tell me a story about a brave knight.",
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 500,
            "thinking_level": "low",
        }
    )

    print(interaction.outputs[-1].text)

### JavaScript

    import { GoogleGenAI } from '@google/genai';

    const client = new GoogleGenAI({});

    const interaction = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: 'Tell me a story about a brave knight.',
        generation_config: {
            temperature: 0.7,
            max_output_tokens: 500,
            thinking_level: 'low',
        }
    });

    console.log(interaction.outputs[interaction.outputs.length - 1].text);

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-3-flash-preview",
        "input": "Tell me a story about a brave knight.",
        "generation_config": {
            "temperature": 0.7,
            "max_output_tokens": 500,
            "thinking_level": "low"
        }
    }'

### Thinking

Gemini 2.5 and newer models use an internal reasoning process called "thinking"
before generating a response. This helps the model produce better answers for
complex tasks like math, coding, and multi-step reasoning.

#### Thinking level

The `thinking_level` parameter lets you control the model's reasoning depth:

| Level | Description | Supported Models |
|---|---|---|
| `minimal` | Matches the "no thinking" setting for most queries. In some cases, models may think very minimally. Minimizes latency and cost. | **Flash Models Only** (e.g. Gemini 3 Flash) |
| `low` | Light reasoning that prioritises latency and cost savings for simple instruction following and chat. | **All Thinking Models** |
| `medium` | Balanced thinking for most tasks. | **Flash Models Only** (e.g. Gemini 3 Flash) |
| `high` | **(Default)** Maximizes reasoning depth. The model may take significantly longer to reach a first token, but the output will be more carefully reasoned. | **All Thinking Models** |

#### Thinking summaries

The model's thinking is represented as **thought blocks** (`type: "thought"`)
in the response outputs. You can control whether to receive human-readable
summaries of the thinking process using the `thinking_summaries` parameter:

| Value | Description |
|---|---|
| `auto` | **(Default)** Returns thought summaries when available. |
| `none` | Disables thought summaries. |

### Python

    from google import genai

    client = genai.Client()

    interaction = client.interactions.create(
        model="gemini-3-flash-preview",
        input="Solve this step by step: What is 15% of 240?",
        generation_config={
            "thinking_level": "high",
            "thinking_summaries": "auto"
        }
    )

    for output in interaction.outputs:
        if output.type == "thought":
            print(f"Thinking: {output.summary}")
        elif output.type == "text":
            print(f"Answer: {output.text}")

### JavaScript

    import { GoogleGenAI } from '@google/genai';

    const client = new GoogleGenAI({});

    const interaction = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: 'Solve this step by step: What is 15% of 240?',
        generation_config: {
            thinking_level: 'high',
            thinking_summaries: 'auto'
        }
    });

    for (const output of interaction.outputs) {
        if (output.type === 'thought') {
            console.log(`Thinking: ${output.summary}`);
        } else if (output.type === 'text') {
            console.log(`Answer: ${output.text}`);
        }
    }

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-3-flash-preview",
        "input": "Solve this step by step: What is 15% of 240?",
        "generation_config": {
            "thinking_level": "high",
            "thinking_summaries": "auto"
        }
    }'

Every thought block contains a `signature` field (a cryptographic hash of the
internal reasoning state) and an optional `summary` field (a human-readable
summary of the model's reasoning). The `signature` is always present, but a
thought block may contain only a signature with no summary in these cases:

- **Simple requests**: The model didn't reason enough to generate a summary
- **`thinking_summaries: "none"`**: Summaries are explicitly disabled

Your code should always handle thought blocks where the `summary` is empty or
absent. When managing conversation history manually (stateless mode), you must
include thought blocks with their signatures in subsequent requests to validate
authenticity.

### Working with files

#### Working with remote files

Access files using remote URLs directly in the API call.  

### Python

    from google import genai
    client = genai.Client()

    interaction = client.interactions.create(
        model="gemini-3-flash-preview",
        input=[
            {
                "type": "image",
                "uri": "https://github.com/<github-path>/cats-and-dogs.jpg",
            },
            {"type": "text", "text": "Describe what you see."}
        ],
    )
    for output in interaction.outputs:
        if output.type == "text":
            print(output.text)

### JavaScript

    import { GoogleGenAI } from '@google/genai';
    const client = new GoogleGenAI({});

    const interaction = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: [
            {
                type: 'image',
                uri: 'https://github.com/<github-path>/cats-and-dogs.jpg',
            },
            { type: 'text', text: 'Describe what you see.' }
        ],
    });
    for (const output of interaction.outputs) {
        if (output.type === 'text') {
            console.log(output.text);
        }
    }

### REST

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-3-flash-preview",
        "input": [
            {
                "type": "image",
                "uri": "https://github.com/<github-path>/cats-and-dogs.jpg"
            },
            {"type": "text", "text": "Describe what you see."}
        ]
    }'

#### Working with Gemini Files API

Upload files to the Gemini [Files API](https://ai.google.dev/gemini-api/docs/files)
before using them.  

### Python

    from google import genai
    import time
    import requests
    client = genai.Client()

    # 1. Download the file
    url = "https://github.com/philschmid/gemini-samples/raw/refs/heads/main/assets/cats-and-dogs.jpg"
    response = requests.get(url)
    with open("cats-and-dogs.jpg", "wb") as f:
        f.write(response.content)

    # 2. Upload to Gemini Files API
    file = client.files.upload(file="cats-and-dogs.jpg")

    # 3. Wait for processing
    while client.files.get(name=file.name).state != "ACTIVE":
        time.sleep(2)

    # 4. Use in Interaction
    interaction = client.interactions.create(
        model="gemini-3-flash-preview",
        input=[
            {
                "type": "image",
                "uri": file.uri,
            },
            {"type": "text", "text": "Describe what you see."}
        ],
    )
    for output in interaction.outputs:
        if output.type == "text":
            print(output.text)

### JavaScript

    import { GoogleGenAI } from '@google/genai';
    import * as fs from 'fs';
    import fetch from 'node-fetch';
    const client = new GoogleGenAI({});

    // 1. Download the file
    const url = 'https://github.com/philschmid/gemini-samples/raw/refs/heads/main/assets/cats-and-dogs.jpg';
    const filename = 'cats-and-dogs.jpg';
    const response = await fetch(url);
    const buffer = await response.buffer();
    fs.writeFileSync(filename, buffer);

    // 2. Upload to Gemini Files API
    const myfile = await client.files.upload({ file: filename, config: { mimeType: 'image/jpeg' } });

    // 3. Wait for processing
    while ((await client.files.get({ name: myfile.name })).state !== 'ACTIVE') {
        await new Promise(resolve => setTimeout(resolve, 2000));
    }

    // 4. Use in Interaction
    const interaction = await client.interactions.create({
        model: 'gemini-3-flash-preview',
        input: [
            { type: 'image', uri: myfile.uri, },
            { type: 'text', text: 'Describe what you see.' }
        ],
    });
    for (const output of interaction.outputs) {
        if (output.type === 'text') {
            console.log(output.text);
        }
    }

### REST

    # 1. Upload the file (Requires File API setup)
    # See https://ai.google.dev/gemini-api/docs/files for details.
    # Assume FILE_URI is obtained from the upload step.

    curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
    -H "Content-Type: application/json" \
    -H "x-goog-api-key: $GEMINI_API_KEY" \
    -d '{
        "model": "gemini-3-flash-preview",
        "input": [
            {"type": "image", "uri": "FILE_URI"},
            {"type": "text", "text": "Describe what you see."}
        ]
    }'

### Data model

You can learn more about the data model in the [API
Reference](https://ai.google.dev/api/interactions-api#data-models). The following is a high
level overview of the main components.

#### Interaction

| Property | Type | Description |
|---|---|---|
| `id` | `string` | Unique identifier for the interaction. |
| `model` / `agent` | `string` | The model or agent used. Only one can be provided. |
| `input` | [`Content[]`](https://ai.google.dev/api/interactions-api#data-models) | The inputs provided. |
| `outputs` | [`Content[]`](https://ai.google.dev/api/interactions-api#data-models) | The model's responses. |
| `tools` | [`Tool[]`](https://ai.google.dev/api/interactions-api#Resource:Tool) | The tools used. |
| `previous_interaction_id` | `string` | ID of the previous interaction for context. |
| `stream` | `boolean` | Whether the interaction is streaming. |
| `status` | `string` | Status: `completed`, `in_progress`, `requires_action`,`failed`, etc. |
| `background` | `boolean` | Whether the interaction is in background mode. |
| `store` | `boolean` | Whether to store the interaction. Default: `true`. Set to `false` to opt out. |
| `usage` | [Usage](https://ai.google.dev/api/interactions-api#Resource:Interaction) | Token usage of the interaction request. |

## Supported models \& agents

| Model Name | Type | Model ID |
|---|---|---|
| Gemini 2.5 Pro | Model | `gemini-2.5-pro` |
| Gemini 2.5 Flash | Model | `gemini-2.5-flash` |
| Gemini 2.5 Flash-lite | Model | `gemini-2.5-flash-lite` |
| Gemini 3 Pro Preview | Model | `gemini-3-pro-preview` |
| Gemini 3 Flash Preview | Model | `gemini-3-flash-preview` |
| Deep Research Preview | Agent | `deep-research-pro-preview-12-2025` |

## How the Interactions API works

The Interactions API is designed around a central resource: the
[**`Interaction`**](https://ai.google.dev/api/interactions-api#Resource:Interaction).
An `Interaction` represents a complete turn in a conversation
or task. It acts as a session record, containing the entire history of an
interaction, including all user inputs, model thoughts, tool calls, tool
results, and final model outputs.

When you make a call to
[`interactions.create`](https://ai.google.dev/api/interactions-api#CreateInteraction), you are
creating a new `Interaction` resource.

### Server-side state management

You can use the `id` of a completed interaction in a subsequent call using the
`previous_interaction_id` parameter to continue the conversation. The server
uses this ID to retrieve the conversation history, saving you from having to
resend the entire chat history.

Only the conversation history (inputs and outputs) is preserved
using `previous_interaction_id`. The other parameters are **interaction-scoped**
and apply only to the specific interaction you are currently generating:

- `tools`
- `system_instruction`
- `generation_config` (including `thinking_level`, `temperature`, etc.)

This means you must re-specify these parameters in each new interaction if you
want them to apply. This server-side state management is optional; you can also
operate in stateless mode by sending the full conversation history in each
request.

### Data storage and retention

By default, all Interaction objects are stored (`store=true`) in order to
simplify use of server-side state management features (with
`previous_interaction_id`), background execution (using `background=true`) and
observability purposes.

- **Paid Tier** : Interactions are retained for **55 days**.
- **Free Tier** : Interactions are retained for **1 day**.

If you do not want this, you can
set `store=false` in your request. This control is separate from state
management; you can opt out of storage for any interaction. However, note that
`store=false` is incompatible with `background=true` and prevents using
`previous_interaction_id` for subsequent turns.

You can delete stored interactions at any time using the delete method found in
the [API Reference](https://ai.google.dev/api/interactions-api). You can only delete interactions if
you know the interaction ID.

After the retention period expires, your data will be
deleted automatically.

Interactions objects are processed according to the [terms](https://ai.google.dev/gemini-api/terms).

## Best practices

- **Cache hit rate** : Using `previous_interaction_id` to continue conversations allows the system to more easily utilize implicit caching for the conversation history, which improves performance and reduces costs.
- **Mixing interactions** : You have the flexibility to mix and match Agent and Model interactions within a conversation. For example, you can use a specialized agent, like the Deep Research agent, for initial data collection, and then use a standard Gemini model for follow-up tasks such as summarizing or reformatting, linking these steps with the `previous_interaction_id`.

## SDKs

You can use latest version of the Google GenAI SDKs in order to access
Interactions API.

- On Python, this is `google-genai` package from `1.55.0` version onwards.
- On JavaScript, this is `@google/genai` package from `1.33.0` version onwards.

You can learn more about how to install the SDKs on
[Libraries](https://ai.google.dev/gemini-api/docs/libraries) page.

## Limitations

- **Beta status**: The Interactions API is in beta/preview. Features and schemas may change.
- **Unsupported features**:
  The following features are not yet supported but are coming soon:

  - [Grounding with Google Maps](https://ai.google.dev/gemini-api/docs/maps-grounding)
- **Output ordering** : Content ordering for built-in tools (`google_search`
  and `url_context`) may sometimes be incorrect, with text appearing before
  the tool execution and result. This is a known issue and a fix is in
  progress.

- **Tool combinations**: Combining MCP, Function Call, and
  Built-in tools is not yet supported but is coming soon.

- **Remote MCP**: Gemini 3 does not support remote mcp, this is coming soon.

## Breaking changes

The Interactions API is currently in an early beta stage. We are actively
developing and refining the API capabilities, resource schemas, and SDK
interfaces based on real-world usage and developer feedback.

As a result, **breaking changes may occur**.
Updates may include changes to:

- Schemas for input and output.
- SDK method signatures and object structures.
- Specific feature behaviors.

For production workloads, you should continue to use the standard
[`generateContent`](https://ai.google.dev/gemini-api/docs/text-generation) API. It remains the
recommended path for stable deployments and will continue to be actively
developed and maintained.

## Feedback

Your feedback is critical to the development of the Interactions API. Please
share your thoughts, report bugs, or request features on our
[Google AI Developer Community Forum](https://discuss.ai.google.dev/c/gemini-api/4).

## What's next

- Try the [Interactions API quickstart notebook](https://colab.sandbox.google.com/github/google-gemini/cookbook/blob/main/quickstarts/Get_started_interactions_api.ipynb).
- Learn more about the [Gemini Deep Research Agent](https://ai.google.dev/gemini-api/docs/deep-research).@google/genai
TypeScript icon, indicating that this package has built-in type declarations
1.40.0 • Public • Published 17 hours ago
Google Gen AI SDK for TypeScript and JavaScript
NPM Downloads Node Current

Documentation: https://googleapis.github.io/js-genai/

The Google Gen AI JavaScript SDK is designed for TypeScript and JavaScript developers to build applications powered by Gemini. The SDK supports both the Gemini Developer API and Vertex AI.

The Google Gen AI SDK is designed to work with Gemini 2.0+ features.

[!CAUTION] API Key Security: Avoid exposing API keys in client-side code. Use server-side implementations in production environments.

Code Generation
Generative models are often unaware of recent API and SDK updates and may suggest outdated or legacy code.

We recommend using our Code Generation instructions codegen_instructions.md when generating Google Gen AI SDK code to guide your model towards using the more recent SDK features. Copy and paste the instructions into your development environment to provide the model with the necessary context.

Prerequisites
Node.js version 20 or later
The following are required for Vertex AI users (excluding Vertex AI Studio)
Select or create a Google Cloud project.

Enable billing for your project.

Enable the Vertex AI API.

Configure authentication for your project.

Install the gcloud CLI.
Initialize the gcloud CLI.
Create local authentication credentials for your user account:
gcloud auth application-default login
A list of accepted authentication options are listed in GoogleAuthOptions interface of google-auth-library-node.js GitHub repo.

Installation
To install the SDK, run the following command:

npm install @google/genai
Quickstart
The simplest way to get started is to use an API key from Google AI Studio:

import {GoogleGenAI} from '@google/genai';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const ai = new GoogleGenAI({apiKey: GEMINI_API_KEY});

async function main() {
  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: 'Why is the sky blue?',
  });
  console.log(response.text);
}

main();
Initialization
The Google Gen AI SDK provides support for both the Google AI Studio and Vertex AI implementations of the Gemini API.

Gemini Developer API
For server-side applications, initialize using an API key, which can be acquired from Google AI Studio:

import { GoogleGenAI } from '@google/genai';
const ai = new GoogleGenAI({apiKey: 'GEMINI_API_KEY'});
Browser
[!CAUTION] API Key Security: Avoid exposing API keys in client-side code. Use server-side implementations in production environments.

In the browser the initialization code is identical:

import { GoogleGenAI } from '@google/genai';
const ai = new GoogleGenAI({apiKey: 'GEMINI_API_KEY'});
Vertex AI
Sample code for VertexAI initialization:

import { GoogleGenAI } from '@google/genai';

const ai = new GoogleGenAI({
    vertexai: true,
    project: 'your_project',
    location: 'your_location',
});
(Optional) (NodeJS only) Using environment variables:
For NodeJS environments, you can create a client by configuring the necessary environment variables. Configuration setup instructions depends on whether you're using the Gemini Developer API or the Gemini API in Vertex AI.

Gemini Developer API: Set GOOGLE_API_KEY as shown below:

export GOOGLE_API_KEY='your-api-key'
Gemini API on Vertex AI: Set GOOGLE_GENAI_USE_VERTEXAI, GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION, as shown below:

export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT='your-project-id'
export GOOGLE_CLOUD_LOCATION='us-central1'
import {GoogleGenAI} from '@google/genai';

const ai = new GoogleGenAI();
API Selection
By default, the SDK uses the beta API endpoints provided by Google to support preview features in the APIs. The stable API endpoints can be selected by setting the API version to v1.

To set the API version use apiVersion. For example, to set the API version to v1 for Vertex AI:

const ai = new GoogleGenAI({
    vertexai: true,
    project: 'your_project',
    location: 'your_location',
    apiVersion: 'v1'
});
To set the API version to v1alpha for the Gemini Developer API:

const ai = new GoogleGenAI({
    apiKey: 'GEMINI_API_KEY',
    apiVersion: 'v1alpha'
});
GoogleGenAI overview
All API features are accessed through an instance of the GoogleGenAI classes. The submodules bundle together related API methods:

ai.models: Use models to query models (generateContent, generateImages, ...), or examine their metadata.
ai.caches: Create and manage caches to reduce costs when repeatedly using the same large prompt prefix.
ai.chats: Create local stateful chat objects to simplify multi turn interactions.
ai.files: Upload files to the API and reference them in your prompts. This reduces bandwidth if you use a file many times, and handles files too large to fit inline with your prompt.
ai.live: Start a live session for real time interaction, allows text + audio + video input, and text or audio output.
Samples
More samples can be found in the github samples directory.

Streaming
For quicker, more responsive API interactions use the generateContentStream method which yields chunks as they're generated:

import {GoogleGenAI} from '@google/genai';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const ai = new GoogleGenAI({apiKey: GEMINI_API_KEY});

async function main() {
  const response = await ai.models.generateContentStream({
    model: 'gemini-2.5-flash',
    contents: 'Write a 100-word poem.',
  });
  for await (const chunk of response) {
    console.log(chunk.text);
  }
}

main();
Function Calling
To let Gemini to interact with external systems, you can provide functionDeclaration objects as tools. To use these tools it's a 4 step

Declare the function name, description, and parametersJsonSchema
Call generateContent with function calling enabled
Use the returned FunctionCall parameters to call your actual function
Send the result back to the model (with history, easier in ai.chat) as a FunctionResponse
import {GoogleGenAI, FunctionCallingConfigMode, FunctionDeclaration, Type} from '@google/genai';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

async function main() {
  const controlLightDeclaration: FunctionDeclaration = {
    name: 'controlLight',
    parametersJsonSchema: {
      type: 'object',
      properties:{
        brightness: {
          type:'number',
        },
        colorTemperature: {
          type:'string',
        },
      },
      required: ['brightness', 'colorTemperature'],
    },
  };

  const ai = new GoogleGenAI({apiKey: GEMINI_API_KEY});
  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: 'Dim the lights so the room feels cozy and warm.',
    config: {
      toolConfig: {
        functionCallingConfig: {
          // Force it to call any function
          mode: FunctionCallingConfigMode.ANY,
          allowedFunctionNames: ['controlLight'],
        }
      },
      tools: [{functionDeclarations: [controlLightDeclaration]}]
    }
  });

  console.log(response.functionCalls);
}

main();
Model Context Protocol (MCP) support (experimental)
Built-in MCP support is an experimental feature. You can pass a local MCP server as a tool directly.

import { GoogleGenAI, FunctionCallingConfigMode , mcpToTool} from '@google/genai';
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

// Create server parameters for stdio connection
const serverParams = new StdioClientTransport({
  command: "npx", // Executable
  args: ["-y", "@philschmid/weather-mcp"] // MCP Server
});

const client = new Client(
  {
    name: "example-client",
    version: "1.0.0"
  }
);

// Configure the client
const ai = new GoogleGenAI({});

// Initialize the connection between client and server
await client.connect(serverParams);

// Send request to the model with MCP tools
const response = await ai.models.generateContent({
  model: "gemini-2.5-flash",
  contents: `What is the weather in London in ${new Date().toLocaleDateString()}?`,
  config: {
    tools: [mcpToTool(client)],  // uses the session, will automatically call the tool using automatic function calling
  },
});
console.log(response.text);

// Close the connection
await client.close();
Generate Content
How to structure contents argument for generateContent
The SDK allows you to specify the following types in the contents parameter:

Content
Content: The SDK will wrap the singular Content instance in an array which contains only the given content instance
Content[]: No transformation happens
Part
Parts will be aggregated on a singular Content, with role 'user'.

Part | string: The SDK will wrap the string or Part in a Content instance with role 'user'.
Part[] | string[]: The SDK will wrap the full provided list into a single Content with role 'user'.
NOTE: This doesn't apply to FunctionCall and FunctionResponse parts, if you are specifying those, you need to explicitly provide the full Content[] structure making it explicit which Parts are 'spoken' by the model, or the user. The SDK will throw an exception if you try this.

Error Handling
To handle errors raised by the API, the SDK provides this ApiError class.

import {GoogleGenAI} from '@google/genai';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const ai = new GoogleGenAI({apiKey: GEMINI_API_KEY});

async function main() {
  await ai.models.generateContent({
    model: 'non-existent-model',
    contents: 'Write a 100-word poem.',
  }).catch((e) => {
    console.error('error name: ', e.name);
    console.error('error message: ', e.message);
    console.error('error status: ', e.status);
  });
}

main();
Interactions (Preview)
Warning: The Interactions API is in Beta. This is a preview of an experimental feature. Features and schemas are subject to breaking changes.

The Interactions API is a unified interface for interacting with Gemini models and agents. It simplifies state management, tool orchestration, and long-running tasks.

See the documentation site for more details.

Basic Interaction
const interaction = await ai.interactions.create({
    model: 'gemini-2.5-flash',
    input: 'Hello, how are you?',
});
console.debug(interaction);
Stateful Conversation
The Interactions API supports server-side state management. You can continue a conversation by referencing the previous_interaction_id.

// 1. First turn
const interaction1 = await ai.interactions.create({
    model: 'gemini-2.5-flash',
    input: 'Hi, my name is Amir.',
});
console.debug(interaction1);

// 2. Second turn (passing previous_interaction_id)
const interaction2 = await ai.interactions.create({
  model: 'gemini-2.5-flash',
  input: 'What is my name?',
  previous_interaction_id: interaction1.id,
});
console.debug(interaction2);
Agents (Deep Research)
You can use specialized agents like deep-research-pro-preview-12-2025 for complex tasks.

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// 1. Start the Deep Research Agent
const initialInteraction = await ai.interactions.create({
  input:
      'Research the history of the Google TPUs with a focus on 2025 and 2026.',
  agent: 'deep-research-pro-preview-12-2025',
  background: true,
});

console.log(`Research started. Interaction ID: ${initialInteraction.id}`);

// 2. Poll for results
while (true) {
  const interaction = await ai.interactions.get(initialInteraction.id);
  console.log(`Status: ${interaction.status}`);

  if (interaction.status === 'completed') {
    console.debug('\nFinal Report:\n', interaction.outputs);
    break;
  } else if (['failed', 'cancelled'].includes(interaction.status)) {
    console.log(`Failed with status: ${interaction.status}`);
    break;
  }

  await sleep(10000);  // Sleep for 10 seconds
}
Multimodal Input
You can provide multimodal data (text, images, audio, etc.) in the input list.

import base64

// Assuming you have a base64 string
// const base64Image = ...;

const interaction = await ai.interactions.create({
  model: 'gemini-2.5-flash',
  input: [
    { type: 'text', text: 'Describe the image.' },
    { type: 'image', data: base64Image, mime_type: 'image/png' },
  ],
});

console.debug(interaction);
Function Calling
You can define custom functions for the model to use. The Interactions API handles the tool selection, and you provide the execution result back to the model.

// 1. Define the tool
const getWeather = (location: string) => {
  /* Gets the weather for a given location. */
  return `The weather in ${location} is sunny.`;
};

// 2. Send the request with tools
let interaction = await ai.interactions.create({
  model: 'gemini-2.5-flash',
  input: 'What is the weather in Mountain View, CA?',
  tools: [
    {
      type: 'function',
      name: 'get_weather',
      description: 'Gets the weather for a given location.',
      parameters: {
        type: 'object',
        properties: {
          location: {
            type: 'string',
            description: 'The city and state, e.g. San Francisco, CA',
          },
        },
        required: ['location'],
      },
    },
  ],
});

// 3. Handle the tool call
for (const output of interaction.outputs!) {
  if (output.type === 'function_call') {
    console.log(
        `Tool Call: ${output.name}(${JSON.stringify(output.arguments)})`);

    // Execute your actual function here
    // Note: ensure arguments match your function signature
    const result = getWeather(JSON.stringify(output.arguments.location));

    // Send result back to the model
    interaction = await ai.interactions.create({
      model: 'gemini-2.5-flash',
      previous_interaction_id: interaction.id,
      input: [
        {
          type: 'function_result',
          name: output.name,
          call_id: output.id,
          result: result,
        },
      ],
    });

    console.debug(`Response: ${JSON.stringify(interaction)}`);
  }
}
Built-in Tools
You can also use Google's built-in tools, such as Google Search or Code Execution.

Grounding with Google Search
const interaction = await ai.interactions.create({
  model: 'gemini-2.5-flash',
  input: 'Who won the last Super Bowl',
  tools: [{ type: 'google_search' }],
});

console.debug(interaction);
Code Execution
const interaction = await ai.interactions.create({
  model: 'gemini-2.5-flash',
  input: 'Calculate the 50th Fibonacci number.',
  tools: [{ type: 'code_execution' }],
});

console.debug(interaction);
Multimodal Output
The Interactions API can generate multimodal outputs, such as images. You must specify the response_modalities.

import * as fs from 'fs';

const interaction = await ai.interactions.create({
  model: 'gemini-3-pro-image-preview',
  input: 'Generate an image of a futuristic city.',
  response_modalities: ['image'],
});

for (const output of interaction.outputs!) {
  if (output.type === 'image') {
    console.log(`Generated image with mime_type: ${output.mime_type}`);
    // Save the image
    fs.writeFileSync(
        'generated_city.png', Buffer.from(output.data!, 'base64'));
  }
}
How is this different from the other Google AI SDKs
This SDK (@google/genai) is Google Deepmind’s "vanilla" SDK for its generative AI offerings, and is where Google Deepmind adds new AI features.

Models hosted either on the Vertex AI platform or the Gemini Developer platform are accessible through this SDK.

Other SDKs may be offering additional AI frameworks on top of this SDK, or may be targeting specific project environments (like Firebase).

The @google/generative_language and @google-cloud/vertexai SDKs are previous iterations of this SDK and are no longer receiving new Gemini 2.0+ features.

Readme
Keywords
none
Package Sidebar
Install
npm i @google/genai


Repository
github.com/googleapis/js-genai

Homepage
github.com/googleapis/js-genai#readme

Weekly Downloads
4,709,757

Version
1.40.0

License
Apache-2.0

Unpacked Size
11.6 MB

Total Files
20

Last publish
17 hours ago

Collaborators
google-wombot
ofrobots
mrdoob
Try on RunKit
Report malware