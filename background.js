chrome.runtime.onInstalled.addListener(() => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const tab = tabs[0];
    if (tab && typeof tab.url === 'string' && !tab.url.startsWith('chrome://')) {
      chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ['onnxruntime-web/ort.js'],
      }, () => {
        chrome.scripting.executeScript({
          target: { tabId: tab.id },
          files: ['content.js'],
        });
      });
    }
  });
});

chrome.runtime.onInstalled.addListener(() => {
  // Create context menu
  chrome.contextMenus.create({
    id: "analyzeImage",
    title: "Check if image is AI-generated",
    contexts: ["image"]
  });
});

// Handle menu click
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "analyzeImage") {
    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: (imageUrl) => {
        const img = document.querySelector(`img[src="${imageUrl}"]`);
        if (img) {
          window.dispatchEvent(new CustomEvent("analyze-specific-image", { detail: img.src }));
        } else {
          console.error("Image not found:", imageUrl);
        }
      },
      args: [info.srcUrl]
    });
  }
});
