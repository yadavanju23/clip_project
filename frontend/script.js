const API_URL = "http://127.0.0.1:8010/search";

const fileInput = document.getElementById("file-input");
const searchBtn = document.getElementById("search-btn");
const preview = document.getElementById("preview");
const resultsEl = document.getElementById("results");
const loadingEl = document.getElementById("loading");
const messageEl = document.getElementById("message");
const dropZone = document.getElementById("drop-zone");
const thresholdSlider = document.getElementById("threshold-slider");
const thresholdValue = document.getElementById("threshold-value");

let selectedFile = null;

function setMessage(message = "") {
  messageEl.textContent = message;
}

function setLoading(isLoading) {
  loadingEl.classList.toggle("hidden", !isLoading);
}

function renderPreview(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    preview.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function renderResults(results) {
  resultsEl.innerHTML = "";

  if (!results.length) {
    setMessage("No results found. Try lowering the similarity threshold.");
    return;
  }

  setMessage(`Found ${results.length} similar images`);

  for (const item of results) {
    const card = document.createElement("div");
    card.className = "result-card";

    const img = document.createElement("img");
    img.src = item.image_url || `../${item.image_path}`;
    img.alt = "Similar result";

    const overlay = document.createElement("div");
    overlay.className = "overlay";
    overlay.textContent = `Similarity: ${(item.similarity * 100).toFixed(1)}%`;

    card.appendChild(img);
    card.appendChild(overlay);
    resultsEl.appendChild(card);
  }
}

async function search() {
  if (!selectedFile) {
    setMessage("Please select an image first.");
    return;
  }

  setLoading(true);
  setMessage("");

  const formData = new FormData();
  formData.append("file", selectedFile);

  const threshold = thresholdSlider.value;

  try {
    const response = await fetch(`${API_URL}?threshold=${threshold}&top_k=20`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      throw new Error(errorBody.detail || "Search request failed");
    }

    const data = await response.json();
    renderResults(data.results || []);
  } catch (error) {
    setMessage(`Error: ${error.message}`);
    resultsEl.innerHTML = "";
  } finally {
    setLoading(false);
  }
}

fileInput.addEventListener("change", (e) => {
  selectedFile = e.target.files[0] || null;
  if (selectedFile) {
    renderPreview(selectedFile);
    setMessage("");
  }
});

searchBtn.addEventListener("click", search);

thresholdSlider.addEventListener("input", () => {
  thresholdValue.textContent = Number(thresholdSlider.value).toFixed(2);
});

["dragenter", "dragover"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropZone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("dragover");
  });
});

dropZone.addEventListener("drop", (e) => {
  const files = e.dataTransfer.files;
  if (files && files[0]) {
    selectedFile = files[0];
    fileInput.files = files;
    renderPreview(selectedFile);
    setMessage("");
  }
});
