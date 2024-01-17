// 1. Import OpenAI library
import OpenAI from "openai";
// 2. Import Pinecone database client
import { Pinecone } from "@pinecone-database/pinecone";
// 3. Import dotenv for environment variable management
import dotenv from "dotenv";
// 4. Load environment variables from .env file
dotenv.config();
// 5. Configuration for Pinecone and OpenAI
const config = {
  similarityQuery: {
    topK: 1, // Top results limit
    includeValues: false, // Exclude vector values
    includeMetadata: true, // Include metadata
  },
  namespace: "your-namespace", // Pinecone namespace
  indexName: "your-index-name", // Pinecone index name
  embeddingID: "your-embedding-id", // Embedding identifier
  dimension: 1536, // Embedding dimension
  metric: "cosine", // Similarity metric
  cloud: "aws", // Cloud provider
  region: "us-west-2", // Serverless region
  query: "What is my dog's name?", // Query example
};
// 6. Data to embed with modified metadata field
const dataToEmbed = [
  {
    textToEmbed: "My dog's name is Steve.",
    favouriteActivities: ["playing fetch", "running in the park"],
    born: "July 19, 2023",
  },
  {
    textToEmbed: "My cat's name is Sandy.",
    favouriteActivities: ["napping", "chasing laser pointers"],
    born: "August 7, 2019",
  },
];
// 7. Initialize OpenAI client with API key
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY, // Optional (as long as OPENAI_API_KEY is in environment variables)
});
// 8. Initialize Pinecone client with API key
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});
// 9. Function to store embeddings in Pinecone
async function storeEmbeddings() {
  // 10. Loop through each data item to embed
  await Promise.all(
    dataToEmbed.map(async (item, index) => {
      // 11. Create embedding using OpenAI
      const embedding = await openai.embeddings.create({
        model: "text-embedding-ada-002",
        input: item.textToEmbed,
      });
      // 12. Define index name and unique ID for each embedding
      const indexName = config.indexName;
      const id = `${config.embeddingID}-${index + 1}`;
      // 13. Upsert embedding into Pinecone with new metadata
      await pc
        .index(indexName)
        .namespace(config.namespace)
        .upsert([
          {
            id: id,
            values: embedding.data[0].embedding,
            metadata: { ...item },
          },
        ]);
      // 14. Log embedding storage
      console.log(`Embedding ${id} stored in Pinecone.`);
    })
  );
}
// 15. Function to query embeddings in Pinecone
async function queryEmbeddings(queryText) {
  // 16. Create query embedding using OpenAI
  const queryEmbedding = await openai.embeddings.create({
    model: "text-embedding-ada-002",
    input: queryText,
  });
  // 17. Perform the query
  const queryResult = await pc
    .index(config.indexName)
    .namespace(config.namespace)
    .query({
      ...config.similarityQuery,
      vector: queryEmbedding.data[0].embedding,
    });
  // 18. Log query results
  console.log(`Query: "${queryText}"`);
  console.log(`Result:`, queryResult);
  console.table(queryResult.matches);
}
// 19. Function to manage Pinecone index
async function manageIndex(action) {
  // 20. Check if index exists
  const indexExists = (await pc.listIndexes()).indexes.some((index) => index.name === config.indexName);
  // 21. Create or delete index based on action
  if (action === "create") {
    if (indexExists) {
      console.log(`Index '${config.indexName}' already exists.`);
    } else {
      await pc.createIndex({
        name: config.indexName,
        dimension: config.dimension,
        metric: config.metric,
        spec: { serverless: { cloud: config.cloud, region: config.region } },
      });
      console.log(`Index '${config.indexName}' created.`);
    }
  } else if (action === "delete") {
    if (indexExists) {
      await pc.deleteIndex(config.indexName);
      console.log(`Index '${config.indexName}' deleted.`);
    } else {
      console.log(`Index '${config.indexName}' does not exist.`);
    }
  } else {
    console.log('Invalid action specified. Use "create" or "delete".');
  }
}
// 22. Main function to orchestrate operations
async function main() {
  await manageIndex("create");
  await storeEmbeddings();
  await queryEmbeddings(config.query);
  // await manageIndex("delete");
}
// 23. Run our main function
main();