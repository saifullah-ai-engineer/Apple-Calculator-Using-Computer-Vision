const statusMessage = document.getElementById("status-message");
const geminiOutput = document.getElementById("gemini-output");
const saveButton = document.getElementById("save-button");
const clearButton = document.getElementById("clear-button");
const solveButton = document.getElementById("solve-button");

function setBusy(button, busy) {
  button.disabled = busy;
}

function setStatus(message) {
  statusMessage.innerText = message;
}

async function postJson(url) {
  const response = await fetch(url, { method: "POST" });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.message || "Request failed.");
  }
  return data;
}

async function saveCanvas() {
  setBusy(saveButton, true);
  setStatus("Saving canvas...");
  try {
    const data = await postJson("/save");
    setStatus(data.message);
  } catch (error) {
    setStatus(error.message);
  } finally {
    setBusy(saveButton, false);
  }
}

async function clearCanvas() {
  setBusy(clearButton, true);
  setStatus("Clearing canvas...");
  try {
    const data = await postJson("/clear");
    geminiOutput.innerText = "Your AI-generated solution will appear here.";
    setStatus(data.message);
  } catch (error) {
    setStatus(error.message);
  } finally {
    setBusy(clearButton, false);
  }
}

async function fetchGeminiOutput() {
  setBusy(solveButton, true);
  setStatus("Solving with Gemini...");
  geminiOutput.innerText = "Thinking...";
  try {
    const data = await postJson("/solve");
    geminiOutput.innerText = data.message;
    setStatus("Solution generated.");
  } catch (error) {
    geminiOutput.innerText = "Unable to generate a solution.";
    setStatus(error.message);
  } finally {
    setBusy(solveButton, false);
  }
}

saveButton.addEventListener("click", saveCanvas);
clearButton.addEventListener("click", clearCanvas);
solveButton.addEventListener("click", fetchGeminiOutput);
