import OpenAI from "openai";

class RecursiveCausalEngine {

    additionalParameters() {
        return super.additionalParameters().concat([
            {
                name: "mainTopics",
                type: "string",
                required: true,
                uiElement: "textarea",
                saveForUser: "local",
                label: "Main Topics",
                description: "Comma-separated list of main variables or topics to explore",
                minHeight: 50
            },
            {
                name: "depth",
                type: "number",
                required: true,
                uiElement: "number",
                saveForUser: "local",
                label: "Depth",
                description: "How many layers of cause/effect to explore"
            }
        ]);
    }

     async generate(prompt, currentModel, parameters) {
        try {
            console.log("here are the params I got", parameters)
            
            const mainTopics = parameters.mainTopics.split(',').map(x => x.trim().toLowerCase()).filter(x => x.length > 0);
            const maxDepth = parameters.depth;

            const explored = new Set();
            const relationships = [];

            async function exploreTopic(topic, depth) {
                console.log("attempting to explore topic", topic, "at depth", depth)
                if (depth > maxDepth || explored.has(topic.toLowerCase())) {
                    console.log("early quitting this exploration")
                    return;
                }

                explored.add(topic.toLowerCase());

                const customPrompt = `Given the following prompt:"""
${prompt}
"""

Identify causes (drivers) and effects (impacts) of the topic: "${topic}".

Return the relationships as a JSON array where each relationship has:
- from: variable
- to: variable
- polarity: + or -
- reasoning: why this relationship exists
- polarityReasoning: why this polarity (+ or -) is appropriate.`;

                const openAIClient = new OpenAI({
                    apiKey: process.env.OPENAI_API_KEY
                });

                const request = {
                    messages: [{
                        role: "user",
                        content: customPrompt 
                    }],
                    model: "gpt-4.1-nano", // fast and cheap, great for hacking together pipelines, not great for best quality
                    temperature: 0,
                };
                console.log("here's exactly what you'll send openai", request)

                const rawResponse = await openAIClient.chat.completions.create(request);

                console.log("here's exactly how openai responded", JSON.stringify(rawResponse, null, 2))

                const response = JSON.parse(rawResponse.choices[0].message.content);
                console.log("the response", response)
                if (response) {
                    console.log("actually got these relationships to stuff into full relationships list", response)
                    relationships.push(...response);

                    const nextTopics = new Set();
                    for (const rel of response) {
                        if (!explored.has(rel.from.toLowerCase())) {
                            console.log("adding new topic to explore", rel.from.toLowerCase())
                            nextTopics.add(rel.from.toLowerCase())
                        }
                        if (!explored.has(rel.to.toLowerCase())) {
                            console.log("adding new topic to explore", rel.to.toLowerCase())
                            nextTopics.add(rel.to.toLowerCase());
                        }
                    }

                    console.log("ripping through these topics", nextTopics)
                    for (const nextTopic of nextTopics) {
                        await exploreTopic(nextTopic, depth + 1);
                    }
                } else {
                    console.log("response came back empty of relationships")
                }
            }

            for (const topic of mainTopics) {
                await exploreTopic(topic, 1);
            }

            const variables = [...new Set(relationships.flatMap(r => [r.from, r.to]))];

            return {
                supportingInfo: {
                    explanation: "Recursive causal relationships extracted up to specified depth.",
                    title: "Recursive Causal Map"
                },
                model: {
                    relationships: relationships,
                    variables: variables
                }
            };

        } catch (err) {
            console.error(err);
            return {
                err: err.toString()
            };
        }
    }
}

export default RecursiveCausalEngine;
