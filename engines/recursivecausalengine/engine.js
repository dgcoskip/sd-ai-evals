// RecursiveCausalEngine.js

import Engine from '../default/engine.js';
import OpenAIWrapper from '../default/OpenAIWrapper.js';

class RecursiveCausalEngine extends Engine {
    constructor() {
        super();
    }

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
            const wrapper = new OpenAIWrapper(parameters);

            const problemStatement = parameters.problemStatement;
            const mainTopics = parameters.mainTopics.split(',').map(x => x.trim()).filter(x => x.length > 0);
            const maxDepth = parameters.depth;

            const explored = new Set();
            const relationships = [];

            async function exploreTopic(topic, depth) {
                if (depth > maxDepth || explored.has(topic.toLowerCase())) {
                    return;
                }

                explored.add(topic.toLowerCase());

                const customPrompt = `Given the following problem statement:\n\n"""\n${problemStatement}\n"""\n\nIdentify causes (drivers) and effects (impacts) of the topic: "${topic}".\n\nReturn the relationships as a JSON array where each relationship has:\n- from: variable\n- to: variable\n- polarity: + or -\n- reasoning: why this relationship exists\n- polarityReasoning: why this polarity (+ or -) is appropriate.`;

                const response = await wrapper.generateDiagram(customPrompt, currentModel);

                if (response.relationships && response.relationships.length > 0) {
                    relationships.push(...response.relationships);

                    const nextTopics = new Set();
                    for (const rel of response.relationships) {
                        if (!explored.has(rel.from.toLowerCase())) {
                            nextTopics.add(rel.from);
                        }
                        if (!explored.has(rel.to.toLowerCase())) {
                            nextTopics.add(rel.to);
                        }
                    }

                    for (const nextTopic of nextTopics) {
                        await exploreTopic(nextTopic, depth + 1);
                    }
                }
                // silently skip if no relationships are found
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
