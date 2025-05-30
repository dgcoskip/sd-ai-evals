import OpenAI from "openai";
import { z } from "zod";
import { zodResponseFormat } from "openai/helpers/zod";

let utils = {};

//this will let us deny old clients in the future
utils.supportedPlatform = function(clientProduct, clientVersion) {
  if (!clientProduct || !clientVersion)
    return false;
  
  //both product and version may be null or undefined if not passed in
  return true;
}

utils.xmileName = function(name) {
  let cleanName = name.replaceAll("\n", " ")
             .replaceAll("\r", " ");

  const splits = cleanName.split(" ").filter((c) => {
    return c !== " ";
  });

  return splits.join("_");
}

utils.caseFold = function(name) {
  let xname = utils.xmileName(name);
  return xname.toLowerCase();
}

utils.convertToXMILE = function(sdJSON) {

  const relationships = sdJSON.relationships;

  let xmileConnectors = "";
  let xmileEqns = "";

  let variablesObj = {}; //variable to causers
  relationships.forEach(function(relationship) {
    if (!variablesObj[relationship.end]) {
      variablesObj[relationship.end] = [];
    }

    let arr = variablesObj[relationship.end];
    if (!arr.includes(relationship.start)) {
      arr.push(relationship.start);
      variablesObj[relationship.end] = arr;

      let polarity = "";
      if (relationship.polarity !== "?")
        polarity =  "polarity=\"" + relationship.polarity + "\"";

      xmileConnectors += "<connector " + polarity + ">";
      xmileConnectors += "<from>" + utils.xmileName(relationship.start) + "</from>";
      xmileConnectors += "<to>" + utils.xmileName(relationship.end) + "</to>";
      xmileConnectors += "</connector>";
    }
  });

  for (const [variable, causers] of Object.entries(variablesObj)) {
    let prettyName = variable.replaceAll("\n", "\\\n").replaceAll("\r", "\\\r");
    xmileEqns += "<aux name=\"" + prettyName + "\">";
    xmileEqns += "<eqn>NAN(";
    causers.forEach(function(cause, index) {
      if (index > 0)
        xmileEqns += ",";
      xmileEqns += utils.xmileName(cause);
    });
    xmileEqns += ")</eqn>";
    xmileEqns += "<isee:delay_aux/>";
    xmileEqns += "</aux>";
  }
  
  let value = '<?xml version="1.0" encoding="utf-8"?>';
  value += '<xmile version="1.0" xmlns="http://docs.oasis-open.org/xmile/ns/XMILE/v1.0" xmlns:isee="http://iseesystems.com/XMILE">';
  value += '<header>';
  value += '<smile version="1.0" namespace="std, isee"/>';
  value += '<vendor>AI Proxy Service</vendor>';
  value += '<product version="1.0.0" lang="en">AI Proxy Service</product>';
  value += '</header>';
  value += '<model>';
  
  value += '<variables>';
  value += xmileEqns;
  value += '</variables>';

  value += '<views>';
  value += '<view type="stock_flow">';
  value += '<style><aux><shape type="name_only"/></aux></style>';
  value += xmileConnectors;
  value += '</view>';
  value += '</views>';
  value += '</model>';
  value += '</xmile>';

  return value;
};

export default utils; 

export const ModelType = Object.freeze({
  GEMINI:   Symbol("Gemini"),
  OPEN_AI:  Symbol("OpenAI"),
  LLAMA: Symbol("Llama"),
  DEEPSEEK: Symbol("Deepseek")
});


export class ModelCapabilities {
  hasStructuredOutput= true;
  hasSystemMode = true;
  hasTemperature = true;
  systemModeUser = 'system';

  name = 'model';

  constructor(modelName) {
      this.name = modelName;

      this.hasStructuredOutput = modelName !== 'o1-mini';
      this.hasSystemMode = modelName !== 'o1-mini';
      this.hasTemperature = !modelName.startsWith('o');
      if (modelName.includes('gemini') || modelName.includes('llama')) {
          this.systemModeUser = 'system';
      } else {
          this.systemModeUser = 'developer';
      }
  }

  get kind() {
      if (this.name.includes('gemini')) {
          return ModelType.GEMINI;
      } else if (this.name.includes('llama')) {
          return ModelType.LLAMA;
      } else if (this.name.includes('deepseek')) {
          return ModelType.DEEPSEEK;
      } else {
          return ModelType.OPEN_AI;
      }
  }
};

export class LLMWrapper {
  #openAIKey;
  #googleKey;
  
  model = new ModelCapabilities(LLMWrapper.DEFAULT_MODEL);
  openAIAPI = null;

  constructor(parameters) {
    if (!parameters.openAIKey) {
        this.#openAIKey = process.env.OPENAI_API_KEY
    } else {
      this.#openAIKey = parameters.openAIKey;
    }

    if (!parameters.googleKey) {
        this.#googleKey = process.env.GOOGLE_API_KEY
    } else {
      this.#googleKey = parameters.googleKey;
    }

    if (parameters.underlyingModel)
      this.model = new ModelCapabilities(parameters.underlyingModel);

    switch (this.model.kind) {
        case ModelType.GEMINI:
            this.openAIAPI = new OpenAI({
                apiKey: this.#googleKey,
                baseURL: "https://generativelanguage.googleapis.com/v1beta/openai"
            });
            break;
        case ModelType.OPEN_AI:
            this.openAIAPI = new OpenAI({
                apiKey: this.#openAIKey,
            });
            break;
        case ModelType.DEEPSEEK:
        case ModelType.LLAMA:
            this.openAIAPI = new OpenAI({
                apiKey: 'junk', // required but unused
                baseURL: 'http://localhost:11434/v1',
            });
            break;
    }
  }

  static SCHEMA_STRINGS = {
    "from": "This is a variable which causes the to variable in this relationship that is between two variables, from and to.  The from variable is the equivalent of a cause.  The to variable is the equivalent of an effect",
    "to": "This is a variable which is impacted by the from variable in this relationship that is between two variables, from and to.  The from variable is the equivalent of a cause.  The to variable is the equivalent of an effect",
    "reasoning": "This is an explanation for why this relationship exists",
    "polarity": "There are two possible kinds of relationships.  The first are relationships with positive polarity that are represented with a + symbol.  In relationships with positive polarity (+) a change in the from variable causes a change in the same direction in the to variable.  For example, in a relationship with postive polarity (+), a decrease in the from variable, would lead to a decrease in the to variable.  The second kind of relationship are those with negative polarity that are represented with a - symbol.  In relationships with negative polarity (-) a change in the from variable causes a change in the opposite direction in the to variable.  For example, in a relationship with negative polarity (-) an increase in the from variable, would lead to a decrease in the to variable.",
    "polarityReasoning": "This is the reason for why the polarity for this relationship was choosen",
    "relationship": "This is a relationship between two variables, from and to (from is the cause, to is the effect).  The relationship also contains a polarity which describes how a change in the from variable impacts the to variable",
    "relationships": "The list of relationships you think are appropriate to satisfy my request based on all of the information I have given you",
    "explanation": "Concisely explain your reasoning for each change you made to the old CLD to create the new CLD. Speak in plain English, don't reference json specifically. Don't reiterate the request or any of these instructions.",
    "title": "A highly descriptive 7 word max title describing your explanation."
};

  static MODELS = [
        {label: "GPT-4o", value: 'gpt-4o'},
        {label: "GPT-4o-mini", value: 'gpt-4o-mini'},
        {label: "GPT-4.5-preview", value: 'gpt-4.5-preview'},
        {label: "GPT-4.1", value: 'gpt-4.1'},
        {label: "GPT-4.1-mini", value: 'gpt-4.1-mini'},
        {label: "GPT-4.1-nano", value: 'gpt-4.1-nano'},
        {label: "Gemini 2.5-flash", value: 'gemini-2.5-flash-preview-04-17'},
        {label: "Gemini 2.5-pro", value: 'gemini-2.5-pro-preview-03-25'},
        {label: "Gemini 2.0", value: 'gemini-2.0-flash'},
        {label: "Gemini 2.0-Lite", value: 'gemini-2.0-flash-lite'},
        {label: "Gemini 1.5", value: 'gemini-1.5-flash'},
        {label: "o1", value: 'o1'},
        {label: "o3-mini low", value: 'o3-mini low'},
        {label: "o3-mini medium", value: 'o3-mini medium'},
        {label: "o3-mini high", value: 'o3-mini high'},
        {label: "o4-mini", value: 'o4-mini'}
    ];

    static DEFAULT_MODEL = 'gpt-4o';

  generateSDJSONResponseSchema() {
      const PolarityEnum = z.enum(["+", "-"]).describe(LLMWrapper.SCHEMA_STRINGS.polarity);

      const Relationship = z.object({
          from: z.string().describe(LLMWrapper.SCHEMA_STRINGS.from),
          to: z.string().describe(LLMWrapper.SCHEMA_STRINGS.to),
          polarity: PolarityEnum,
          reasoning: z.string().describe(LLMWrapper.SCHEMA_STRINGS.reasoning),
          polarityReasoning: z.string().describe(LLMWrapper.SCHEMA_STRINGS.polarityReasoning)
      }).describe(LLMWrapper.SCHEMA_STRINGS.relationship);
          
      const Relationships = z.object({
          explanation: z.string().describe(LLMWrapper.SCHEMA_STRINGS.explanation),
          title: z.string().describe(LLMWrapper.SCHEMA_STRINGS.title),
          relationships: z.array(Relationship).describe(LLMWrapper.SCHEMA_STRINGS.relationships)
      });

      return zodResponseFormat(Relationships, "relationships_response");
  }

  static additionalParameters() {
    return [{
            name: "openAIKey",
            type: "string",
            required: true,
            uiElement: "password",
            saveForUser: "global",
            label: "Open AI API Key",
            description: "Leave blank for the default, or your Open AI key - skprojectXXXXX"
        },{
            name: "googleKey",
            type: "string",
            required: true,
            uiElement: "password",
            saveForUser: "global",
            label: "Google API Key",
            description: "Leave blank for the default, or your Google API key - XXXXXX"
        },{
            name: "underlyingModel",
            type: "string",
            defaultValue: LLMWrapper.DEFAULT_MODEL,
            required: false,
            options: LLMWrapper.MODELS,
            uiElement: "combobox",
            saveForUser: "local",
            label: "LLM Model",
            description: "The LLM model that you want to use to process your queries."
        }];
    }
};

