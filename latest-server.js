const express = require("express");
const fs = require("fs/promises");
const path = require("path");
const { existsSync, mkdirSync } = require("fs");
// We'll use fs.readdir instead of glob
const crypto = require("crypto");
const { GoogleGenAI, Modality } = require("@google/genai");

const app = express();
app.use(express.json());

// --- UPDATE THIS if your new workflow file has a different name ---
const WORKFLOW_FILE = "workflow.json";
// ---

const OUTPUT_DIR = "./output";
const COMFY_API_URL = "http://127.0.0.1:8188"; // Base URL for ComfyUI API

// Ensure output directory exists
if (!existsSync(OUTPUT_DIR)) {
  mkdirSync(OUTPUT_DIR, { recursive: true });
}

const apiKey = process.env.GEMINI_API_KEY;

/**
 * Generate a random client ID using built-in libraries
 * @param {number} length - Length of the client ID
 * @returns {string} - Random client ID
 */
function generateClientId(length = 10) {
  return crypto
    .randomBytes(Math.ceil(length / 2))
    .toString("hex")
    .slice(0, length);
}

/**
 * Fetches an image from a URL and converts it to base64
 * @param {string} imageUrl - URL of the image to fetch
 * @returns {Promise<string>} - Base64 encoded image data
 */
async function fetchImageAsBase64(imageUrl) {
  const response = await fetch(imageUrl);
  const arrayBuffer = await response.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);
  return buffer.toString("base64");
}

/**
 * Helper function to make API requests
 * @param {string} url - API endpoint
 * @param {string} method - HTTP method
 * @param {object} data - Request data
 * @returns {Promise<object>} - Response object
 */
async function makeApiRequest(url, method = "GET", data = null) {
  const fullUrl = `${COMFY_API_URL}${url}`;
  const options = {
    method,
    headers: data ? { "Content-Type": "application/json" } : {},
  };

  if (data) {
    options.body = JSON.stringify(data);
  }

  try {
    const response = await fetch(fullUrl, options);
    let content = null;

    try {
      // Try to parse as JSON first
      const text = await response.text();
      content = text ? JSON.parse(text) : null;
    } catch (error) {
      // If JSON parsing fails, use the raw text
      const text = await response.text();
      content = text || null;
    }

    return {
      status_code: response.status,
      content,
    };
  } catch (error) {
    console.error(`Error calling ${fullUrl}: ${error.message}`);

    if (error.name === "TypeError" && error.message.includes("fetch failed")) {
      return {
        status_code: 503,
        error: `Could not connect to ComfyUI: ${error.message}`,
      };
    }

    return {
      status_code: 500,
      error: error.message,
    };
  }
}

/**
 * Health check endpoint to verify the API is running and ComfyUI is accessible
 */
app.get("/health", async (req, res) => {
  let comfyStatus = "disconnected";
  let comfyError = null;

  try {
    const comfyResponse = await makeApiRequest("/system_stats");
    if (comfyResponse.status_code === 200) {
      comfyStatus = "connected";
    } else {
      comfyError = comfyResponse.error || "Unknown ComfyUI connection issue";
      console.log(`Health Check - ComfyUI Error: ${comfyError}`);
    }
  } catch (error) {
    comfyError = error.message;
    console.log(
      `Health Check - Exception connecting to ComfyUI: ${comfyError}`
    );
  }

  const response = {
    status: "healthy",
    message: "API is running",
    comfy_ui: comfyStatus,
  };

  if (comfyError) {
    response.comfy_ui_error = comfyError;
  }

  return res.status(200).json(response);
});

/**
 * Run YOLO detection using the updated ComfyUI workflow and return text output.
 * Expects 'folderName', 'imageUrl', and 'categories' as JSON parameters.
 */
app.post("/detect", async (req, res) => {
  // Get parameters from request JSON
  const data = req.body;
  if (!data) {
    return res.status(400).json({ error: "Missing JSON request body" });
  }

  const folderName = data.folderName;
  const imageUrl = data.imageUrl;
  const categories = data.categories;

  const missingParams = [];
  if (!folderName) missingParams.push("folderName");
  if (!imageUrl) missingParams.push("imageUrl");
  if (!categories) missingParams.push("categories");

  if (missingParams.length > 0) {
    return res.status(400).json({
      error: `Missing required parameters: ${missingParams.join(", ")}`,
    });
  }

  // --- Workflow Processing ---
  // Create specific output folder
  const folderPath = path.join(OUTPUT_DIR, folderName);
  try {
    await fs.mkdir(folderPath, { recursive: true });
    console.log(`Output will be saved to: ${folderPath}`);
  } catch (error) {
    console.error(`Error creating directory ${folderPath}: ${error.message}`);
    return res
      .status(500)
      .json({ error: `Failed to create output directory: ${error.message}` });
  }

  // Load the workflow
  let workflow;
  try {
    const workflowContent = await fs.readFile(WORKFLOW_FILE, "utf8");
    workflow = JSON.parse(workflowContent);
  } catch (error) {
    if (error.code === "ENOENT") {
      console.error(`Error: Workflow file not found at ${WORKFLOW_FILE}`);
      return res.status(500).json({
        error: `Workflow file '${WORKFLOW_FILE}' not found on server.`,
      });
    } else if (error instanceof SyntaxError) {
      console.error(
        `Error: Failed to parse workflow file ${WORKFLOW_FILE}: ${error.message}`
      );
      return res
        .status(500)
        .json({ error: `Invalid JSON in workflow file '${WORKFLOW_FILE}'.` });
    } else {
      console.error(
        `Error: Failed to load workflow ${WORKFLOW_FILE}: ${error.message}`
      );
      return res
        .status(500)
        .json({ error: `Failed to load workflow: ${error.message}` });
    }
  }

  // --- Modify the workflow ---
  // --- Update these node IDs based on the NEW workflow ---
  const nodesToModify = new Set(["10", "14", "16", "18", "21"]); // Node IDs for Yoloworld, LoadImageFromURL, SaveTextFile
  // ---

  // Check if required nodes exist in the loaded workflow
  const loadedNodeIds = new Set(Object.keys(workflow));
  const missingNodes = [...nodesToModify].filter(
    (node) => !loadedNodeIds.has(node)
  );

  if (missingNodes.length > 0) {
    console.error(`Error: Workflow missing required nodes: ${missingNodes}`);
    return res.status(500).json({
      error: `Workflow file '${WORKFLOW_FILE}' is missing required nodes: ${missingNodes.join(
        ", "
      )}`,
    });
  }

  try {
    // --- Update node modification logic for NEW workflow ---
    // Modify Save Text File nodes for output path
    workflow["16"].inputs.path = folderPath;
    console.log(
      `Workflow node '16' (Save Text File) path set to: ${folderPath}`
    );

    // Also modify node 21 (Save Text File) if it exists
    if ("21" in workflow) {
      workflow["21"].inputs.path = folderPath;
      console.log(
        `Workflow node '21' (Save Text File) path set to: ${folderPath}`
      );
    }

    // Modify Load Image From Url node (14) for image URL
    workflow["14"].inputs.urls = imageUrl;
    console.log(
      `Workflow node '14' (Load Image From Url) urls set to: ${imageUrl}`
    );

    // Also modify node 17 (Load Image From Url) if it exists
    if ("17" in workflow) {
      workflow["17"].inputs.urls = imageUrl;
      console.log(
        `Workflow node '17' (Load Image From Url) urls set to: ${imageUrl}`
      );
    }

    // Modify Yoloworld ESAM node (10) for categories
    workflow["10"].inputs.categories = categories;
    console.log(
      `Workflow node '10' (Yoloworld ESAM) categories set to: ${categories}`
    );

    // Also modify node 18 (Yoloworld ESAM) if it exists
    if ("18" in workflow) {
      workflow["18"].inputs.categories = categories;
      console.log(
        `Workflow node '18' (Yoloworld ESAM) categories set to: ${categories}`
      );
    }
    // ---
  } catch (error) {
    if (error instanceof TypeError) {
      console.error(
        `Error: Missing 'inputs' or specific key in workflow node. Check if workflow structure matches expected format.`
      );
      return res.status(500).json({
        error: `Workflow structure error: Missing key in node inputs.`,
      });
    } else {
      console.error(`Error modifying workflow: ${error.message}`);
      return res
        .status(500)
        .json({ error: `Internal error modifying workflow: ${error.message}` });
    }
  }

  // Generate a unique client ID
  const clientId = generateClientId();

  // --- Queue the prompt in ComfyUI ---
  console.log(`Queueing prompt with client_id: ${clientId}`);
  const queueResponse = await makeApiRequest("/prompt", "POST", {
    prompt: workflow,
    client_id: clientId,
  });

  if (queueResponse.status_code !== 200) {
    const errorMsg = queueResponse.error || "Unknown error";
    console.error(
      `Failed to queue workflow. Status: ${queueResponse.status_code}. Error: ${errorMsg}`
    );
    return res
      .status(500)
      .json({ error: `Failed to queue workflow: ${errorMsg}` });
  }

  const promptId = queueResponse.content?.prompt_id;
  if (!promptId) {
    console.error(
      "Error: No prompt_id returned from ComfyUI /prompt endpoint."
    );
    return res
      .status(500)
      .json({ error: "No prompt ID received from ComfyUI after queuing." });
  }

  console.log(`Workflow queued successfully. Prompt ID: ${promptId}`);

  // --- Wait for the workflow to complete ---
  const maxWaitTime = 120; // Timeout in seconds
  const startTime = Date.now();
  let workflowCompleted = false;
  let finalHistory = null;

  console.log(`Polling ComfyUI history for prompt_id: ${promptId}...`);

  while ((Date.now() - startTime) / 1000 < maxWaitTime) {
    const historyResponse = await makeApiRequest(`/history/${promptId}`);

    if (historyResponse.status_code === 200) {
      const historyContent = historyResponse.content;
      if (historyContent && promptId in historyContent) {
        finalHistory = historyContent[promptId];
        workflowCompleted = true;
        console.log(`Workflow ${promptId} completed.`);
        break;
      }
    } else if (historyResponse.status_code === 404) {
      // Still waiting, keep polling
    } else {
      const errorMsg = historyResponse.error || "Unknown polling error";
      console.error(
        `Error polling history for ${promptId}. Status: ${historyResponse.status_code}. Error: ${errorMsg}`
      );
      if (historyResponse.status_code >= 500) {
        return res
          .status(500)
          .json({ error: `Error checking workflow status: ${errorMsg}` });
      }
    }

    // Wait before polling again
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }

  if (!workflowCompleted) {
    console.error(
      `Workflow execution timed out after ${maxWaitTime} seconds for prompt_id: ${promptId}`
    );
    return res.status(500).json({ error: "Workflow execution timed out" });
  }

  // --- Retrieve the output file ---
  // Brief pause for filesystem sync if needed
  await new Promise((resolve) => setTimeout(resolve, 1000));

  try {
    // Use the specific folderPath created earlier where Node '16' and '21' saved the file
    const files = await fs.readdir(folderPath);
    const txtFiles = files
      .filter((file) => file.endsWith(".txt"))
      .map((file) => path.join(folderPath, file));

    if (txtFiles.length === 0) {
      console.error(
        `Error: No output .txt file found in ${folderPath} for prompt_id ${promptId}`
      );

      // Check if we have any outputs in the history
      if (finalHistory && "outputs" in finalHistory) {
        console.log(
          `ComfyUI history outputs for ${promptId}: ${JSON.stringify(
            finalHistory.outputs,
            null,
            2
          )}`
        );
      }

      // If no outputs at all, return error
      console.log(
        `ComfyUI history outputs for ${promptId}: ${JSON.stringify(
          finalHistory?.outputs || "No outputs found",
          null,
          2
        )}`
      );

      return res.status(500).json({
        error: "No output text file was generated by the workflow",
        note: "Check if Save Text File nodes (16 and 21) are properly configured in the workflow",
        workflow_outputs: finalHistory?.outputs || null,
      });
    }

    // Sort files by creation time to get the latest one
    const fileStats = await Promise.all(
      txtFiles.map(async (file) => ({
        file,
        stat: await fs.stat(file),
      }))
    );

    const latestFile = fileStats
      .sort((a, b) => b.stat.ctimeMs - a.stat.ctimeMs)
      .map((entry) => entry.file)[0];

    console.log(`Found output text file: ${latestFile}`);

    const fileContent = await fs.readFile(latestFile, "utf-8");

    // Return success response with the file content
    return res.status(200).json({
      status: "success",
      output_content: fileContent,
      file_path: latestFile, // This is the path on the *server*
    });
  } catch (error) {
    console.error(
      `Error accessing output file in ${folderPath}: ${error.message}`
    );
    return res
      .status(500)
      .json({ error: `Failed to read output file: ${error.message}` });
  }
});

// Add a general error handler for the Express app
app.use((err, req, res, next) => {
  console.error(`Unhandled error: ${err.message}`);
  console.error(err.stack);
  res.status(500).json({
    error: "Internal server error",
    message: err.message,
  });
});

/**
 * Generate an image using Gemini API based on an image URL and prompt
 * Expects 'imageUrl' and 'prompt' as JSON parameters.
 */
app.post("/generate", async (req, res) => {
  // Extract parameters from request JSON
  const data = req.body;
  if (!data) {
    return res.status(400).json({ error: "Missing JSON request body" });
  }

  const imageUrl = data.imageUrl;
  const prompt = data.prompt;

  const missingParams = [];
  if (!imageUrl) missingParams.push("imageUrl");
  if (!prompt) missingParams.push("prompt");

  if (missingParams.length > 0) {
    return res.status(400).json({
      error: `Missing required parameters: ${missingParams.join(", ")}`,
    });
  }

  try {
    // Initialize Gemini API
    const ai = new GoogleGenAI({
      apiKey, // Should be in environment variables in production
    });

    // Fetch and convert image
    console.log(`Fetching image from URL: ${imageUrl}`);
    const base64Image = await fetchImageAsBase64(imageUrl);

    // Prepare content parts
    const contents = [
      { text: prompt },
      {
        inlineData: {
          mimeType: "image/png", // We might need to detect the actual mime type
          data: base64Image,
        },
      },
    ];

    console.log(`Calling Gemini API with prompt: "${prompt}"`);
    // Call Gemini API
    const response = await ai.models.generateContent({
      model: "gemini-2.0-flash-exp-image-generation",
      contents: contents,
      config: {
        responseModalities: [Modality.TEXT, Modality.IMAGE],
      },
    });

    // Process response
    const result = {
      status: "success",
      text: null,
      imageData: null,
    };

    for (const part of response.candidates[0].content.parts) {
      if (part.text) {
        result.text = part.text;
      } else if (part.inlineData) {
        result.imageData = part.inlineData.data;
        console.log("Received image data from Gemini API");
      }
    }

    return res.status(200).json(result);
  } catch (error) {
    console.error(`Error in /generate endpoint: ${error.message}`);
    return res.status(500).json({
      error: "Failed to generate image",
      details: error.message,
    });
  }
});

/**
 * Process image with Gemini to identify furniture and items
 * @param {string} imageUrl - URL of the image to process
 * @returns {Promise<object>} - Identified items or error
 */
async function processImageWithGemini(imageUrl) {
  try {
    // Fetch the image as base64 using the existing function
    const base64Image = await fetchImageAsBase64(imageUrl);

    // Initialize Gemini API (already imported at the top)
    const ai = new GoogleGenAI({
      apiKey, // Should be in environment variables in production
    });

    // Prepare content parts
    const contents = [
      {
        text: "give me furniture and items (except walls) in a json array. Use only the basic name of each item (e.g., 'chair', 'lamp', 'table'). Output should be an array of strings.",
      },
      {
        inlineData: {
          mimeType: "image/jpeg", // We might need to detect the actual mime type
          data: base64Image,
        },
      },
    ];

    // Call Gemini API
    const response = await ai.models.generateContent({
      model: "gemini-2.0-flash-exp-image-generation",
      contents: contents,
      config: {
        responseModalities: [Modality.TEXT],
      },
    });

    // Extract and parse the response
    const textResponse = response.candidates[0].content.parts[0].text;
    try {
      // Try to parse the response as JSON
      return JSON.parse(textResponse);
    } catch (parseError) {
      // If parsing fails, try to extract JSON from the text
      const jsonMatch = textResponse.match(/\[[\s\S]*?\]/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }
      // If all parsing attempts fail, return the raw text
      return { error: "Failed to parse response", rawResponse: textResponse };
    }
  } catch (error) {
    console.error("Error processing image with Gemini:", error);
    throw new Error(`Failed to process image: ${error.message}`);
  }
}

/**
 * Process an image with Gemini to identify furniture and items
 * Expects 'imageUrl' as JSON parameter.
 */
app.post("/items", async (req, res) => {
  // Extract parameters from request JSON
  const data = req.body;
  if (!data) {
    return res.status(400).json({ error: "Missing JSON request body" });
  }

  const imageUrl = data.imageUrl;

  if (!imageUrl) {
    return res.status(400).json({
      error: "Missing required parameter: imageUrl",
    });
  }

  try {
    // Call our local implementation of processImageWithGemini
    const result = await processImageWithGemini(imageUrl);

    return res.status(200).json({
      status: "success",
      items: result,
    });
  } catch (error) {
    console.error(`Error in /items endpoint: ${error.message}`);
    return res.status(500).json({
      error: "Failed to process image",
      details: error.message,
    });
  }
});

// Handle uncaught exceptions to prevent the server from crashing
process.on("uncaughtException", (err) => {
  console.error("Uncaught Exception:");
  console.error(err.stack);
  // Keep the process running despite the error
});

process.on("unhandledRejection", (reason, promise) => {
  console.error("Unhandled Rejection at:", promise);
  console.error("Reason:", reason);
  // Keep the process running despite the rejection
});

// Start the server
const PORT = 5050;
const server = app.listen(PORT, "0.0.0.0", () => {
  console.log(`Starting Express server on port ${PORT}...`);
  console.log(`Using ComfyUI API at: ${COMFY_API_URL}`);
  console.log(`Loading workflow from: ${WORKFLOW_FILE}`);
  console.log(`Saving outputs to subdirectories within: ${OUTPUT_DIR}`);
});

// Handle server errors
server.on("error", (err) => {
  console.error(`Server error: ${err.message}`);
  if (err.code === "EADDRINUSE") {
    console.error(
      `Port ${PORT} is already in use. Please use a different port.`
    );
  }
});
