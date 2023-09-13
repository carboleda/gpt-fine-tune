import { OpenAI } from "langchain/llms/openai";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { CSVLoader } from "langchain/document_loaders/fs/csv";
import {
  VectorStoreToolkit,
  createVectorStoreAgent,
  VectorStoreInfo,
} from "langchain/agents";

console.log(Bun.env.OPENAI_API_KEY);
const model = new OpenAI(
  {}
  // {
  //   defaultHeaders: {
  //     "OpenAI-Organization": "org-FjCQ3ZGWqJj9j6VkAq8RaBcU",
  //   },
  // }
);

console.log("Loading files...");
const loader = new DirectoryLoader("./data", {
  //   ".json": (path) => new JSONLoader(path, "/texts"),
  //   ".jsonl": (path) => new JSONLinesLoader(path, "/html"),
  ".txt": (path) => new TextLoader(path),
  ".csv": (path) => new CSVLoader(path, "text"),
});
const docs = await loader.load();
console.log("docs", docs);

/* Create the vectorstore */
const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());

/* Create the agent */
const vectorStoreInfo: VectorStoreInfo = {
  name: "state_of_union_address",
  description: "the most recent state of the Union address",
  vectorStore,
};

const toolkit = new VectorStoreToolkit(vectorStoreInfo, model);
const agent = createVectorStoreAgent(model, toolkit);

const input = "what is my dog's name";
console.log(`Executing: ${input}`);

const result = await agent.call({ input });
console.log(`Got output ${result.output}`);
