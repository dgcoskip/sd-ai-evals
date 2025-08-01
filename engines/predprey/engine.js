class Engine {
    constructor() {

    }

    static description() {
        return `Demonstration engine that doesn't use AI and always returns the same simple predator-prey balancing loop. Typically used for testing and understanding SD-AI structure.`;
    }

    static supportedModes() {
        return ["cld"];
    }

    additionalParameters () {
        return []
    } 

    async generate(prompt, currentModel, parameters) {
        return {
            model: {
                variables: [
                    {
                        name: "predator",
                        type: "variable"
                    },
                    {
                        name: "prey",
                        type: "variable"
                    }
                ],
                relationships: [
                    {
                        from: "predator",
                        to: "prey",
                        polarity: "-",
                        reasoning: "As the number of predators increases the number of prey decreases because they get eaten.",
                        polarityReasoning: "The polarity is - because as predators goes up prey goes down"
                    },
                    {
                        from: "prey",
                        to: "predator",
                        polarity: "+",
                        reasoning: "As the number of prey increases the number of predators increase because they have more food.",
                        polarityReasoning: "The polarity is + because as prey goes up predators goes up"
                    }
                ]
            }
        }
    }
}

export default Engine;