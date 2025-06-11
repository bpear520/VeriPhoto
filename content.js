console.log("Content script loaded.");

// Ensure `ort` is defined since `ort.js` is now statically loaded
if (typeof ort === "undefined") {
  console.error("ONNX Runtime Web is not loaded.");
} else {
  console.log("ONNX Runtime Web is available globally:", ort);

  // Set the log level for ONNX Runtime to verbose for detailed debugging information
  ort.env.logLevel = "verbose"; // Options: 'verbose', 'info', 'warning', 'error'

  // Set up the path for WebAssembly files
  ort.env.wasm.wasmPaths = chrome.runtime.getURL("onnxruntime-web/");

  const wasmLoaderUrl = chrome.runtime.getURL(
    "onnxruntime-web/ort-wasm-simd-threaded.mjs"
  );
  import(wasmLoaderUrl)
    .then(() => {
      console.log("WASM module loaded successfully.");

      // Load ONNX model and run inference
      loadModelAndRunInference();
    })
    .catch((error) => {
      console.error("Failed to load WASM module:", error);
    });
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// Function to load the model and process images
async function loadModelAndRunInference() {
  window.addEventListener("analyze-specific-image", async (event) => {
    const imageUrl = event.detail;
    const img = document.querySelector(`img[src="${imageUrl}"]`);
    if (img && typeof ort !== "undefined") {
      const session = await ort.InferenceSession.create(
        chrome.runtime.getURL("model.onnx")
      );
      processImage(img, session);
    }
  });
}

// Function to process a single image
function processImage(imgElement, session) {
  console.log("Processing image:", imgElement.src);

  const img = new Image();
  img.crossOrigin = "Anonymous";
  img.src = imgElement.src;

  img.onload = async () => {
    const imageData = getImageData(img);

    if (imageData) {
      console.log("Image data prepared for inference.");
      const isDeepfake = await runInference(session, imageData);
      console.log("Inference result:", isDeepfake);

      if (isDeepfake) {
        imgElement.style.border = "5px solid red";
        console.log("Deepfake detected for this image.");
      } else {
        console.log("This image is not a deepfake.");
      }
    } else {
      console.error("Failed to get image data for:", imgElement.src);
    }
  };

  img.onerror = (error) => {
    console.error("Error loading image:", error);
  };
}

// Function to get image data from the DOM
function getImageData(imgElement) {
  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext("2d");

  imgElement.crossOrigin = "Anonymous";

  ctx.drawImage(imgElement, 0, 0, 224, 224);

  try {
    const imageData = ctx.getImageData(0, 0, 224, 224);
    console.log("Successfully got image data for:", imgElement.src);
    console.log(
      "Raw Image Data (first 10 values):",
      imageData.data.slice(0, 10)
    );
    return imageData;
  } catch (e) {
    console.error("Error accessing canvas data for:", imgElement.src, e);
    return null;
  }
}

// Function to run inference using the ONNX model
async function runInference(session, imageData) {
  try {
    const inputTensor = preprocessImageData(imageData);

    const feeds = { input: inputTensor };
    const output = await session.run(feeds);

    const outputName = session.outputNames[0];
    console.log("Model input names:", session.inputNames);
    console.log("Model output names:", session.outputNames);

    const logits = output[outputName].data;

    const logit = logits[0];
    const realProbability = sigmoid(logit);
    const fakeProbability = 1 - realProbability;

    console.log("Model output logit:", logit);
    console.log("Probability of being real:", realProbability);
    console.log("Probability of being fake:", fakeProbability);

    return fakeProbability > 0.5;
  } catch (error) {
    console.error("Error during inference:", error);
    return null;
  }
}

// Function to preprocess image data for ONNX model
function preprocessImageData(imageData) {
  const width = imageData.width;
  const height = imageData.height;
  const data = imageData.data;

  const mean = [0.485, 0.456, 0.406];  // RGB means
  const std  = [0.229, 0.224, 0.225];  // RGB std devs

  const floatData = new Float32Array(1 * 3 * height * width);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const r = data[idx * 4 + 0] / 255.0;
      const g = data[idx * 4 + 1] / 255.0;
      const b = data[idx * 4 + 2] / 255.0;

      // Normalize using ImageNet stats
      floatData[0 * height * width + y * width + x] = (r - mean[0]) / std[0]; // R
      floatData[1 * height * width + y * width + x] = (g - mean[1]) / std[1]; // G
      floatData[2 * height * width + y * width + x] = (b - mean[2]) / std[2]; // B
    }
  }

  return new ort.Tensor('float32', floatData, [1, 3, height, width]);
}

