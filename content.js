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
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");

  imgElement.crossOrigin = "Anonymous";

  ctx.drawImage(imgElement, 0, 0, 256, 256);

  try {
    const imageData = ctx.getImageData(0, 0, 256, 256);
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
    const { rgbTensor, noiseTensor } = preprocessImageData(imageData);

    // List input names for clarity
    const inputNames = session.inputNames;
    console.log("Model input names:", inputNames);  // Should include 'rgb_input' and 'noise_input'

    const feeds = {};
    feeds[inputNames[0]] = rgbTensor;     // e.g. 'rgb_input'
    feeds[inputNames[1]] = noiseTensor;   // e.g. 'noise_input'

    const output = await session.run(feeds);

    const outputName = session.outputNames[0];
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

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];

  const rgb = new Float32Array(1 * 3 * height * width);
  const noise = new Float32Array(1 * 3 * height * width);

  // Precompute grayscale version for Laplacian (simple luminance approximation)
  const grayscale = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const r = data[idx * 4 + 0] / 255.0;
      const g = data[idx * 4 + 1] / 255.0;
      const b = data[idx * 4 + 2] / 255.0;
      grayscale[idx] = 0.2989 * r + 0.5870 * g + 0.1140 * b;
    }
  }

  // Apply 3x3 Laplacian kernel to grayscale image
  const laplacian = new Float32Array(width * height);
  const kernel = [
     0, -1,  0,
    -1,  4, -1,
     0, -1,  0
  ];

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let sum = 0.0;
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const ix = x + kx;
          const iy = y + ky;
          const w = kernel[(ky + 1) * 3 + (kx + 1)];
          sum += w * grayscale[iy * width + ix];
        }
      }
      laplacian[y * width + x] = sum;
    }
  }

  // Populate RGB and noise tensors
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;
      const r = data[idx * 4 + 0] / 255.0;
      const g = data[idx * 4 + 1] / 255.0;
      const b = data[idx * 4 + 2] / 255.0;

      const rNorm = (r - mean[0]) / std[0];
      const gNorm = (g - mean[1]) / std[1];
      const bNorm = (b - mean[2]) / std[2];

      rgb[0 * height * width + y * width + x] = rNorm;
      rgb[1 * height * width + y * width + x] = gNorm;
      rgb[2 * height * width + y * width + x] = bNorm;

      const n = laplacian[idx];  // noise scalar for all 3 channels
      noise[0 * height * width + y * width + x] = n;
      noise[1 * height * width + y * width + x] = n;
      noise[2 * height * width + y * width + x] = n;
    }
  }

  return {
    rgbTensor: new ort.Tensor('float32', rgb, [1, 3, height, width]),
    noiseTensor: new ort.Tensor('float32', noise, [1, 3, height, width])
  };
}
