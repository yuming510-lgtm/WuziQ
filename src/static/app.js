const boardElement = document.getElementById("board");
const statusElement = document.getElementById("status-text");
const messageElement = document.getElementById("message");
const undoButton = document.getElementById("undo-btn");
const resetButton = document.getElementById("reset-btn");
const sizeInput = document.getElementById("size-input");
const aiSelect = document.getElementById("ai-select");

let currentState = null;

async function fetchState() {
  try {
    const response = await fetch("/state");
    if (!response.ok) {
      throw new Error(`Failed to fetch state: ${response.status}`);
    }
    const state = await response.json();
    currentState = state;
    renderState(state);
    clearMessage();
  } catch (error) {
    showMessage(error.message, true);
  }
}

function renderState(state) {
  const size = state.size;
  statusElement.textContent = buildStatusText(state);
  sizeInput.value = size;
  aiSelect.value = state.ai_side;

  boardElement.style.setProperty("--board-size", size);
  boardElement.innerHTML = "";

  for (let row = 0; row < size; row += 1) {
    for (let col = 0; col < size; col += 1) {
      const cell = document.createElement("button");
      cell.type = "button";
      cell.className = "cell";
      cell.dataset.row = row;
      cell.dataset.col = col;
      const value = state.board[row][col];
      if (value === "B") {
        cell.classList.add("black");
        cell.textContent = "●";
      } else if (value === "W") {
        cell.classList.add("white");
        cell.textContent = "○";
      }
      if (state.last_move && state.last_move.row === row && state.last_move.col === col) {
        cell.classList.add("last-move");
      }
      if (value !== null || state.winner) {
        cell.disabled = true;
      }
      cell.addEventListener("click", handleCellClick);
      boardElement.appendChild(cell);
    }
  }
}

function buildStatusText(state) {
  if (state.winner === "draw") {
    return "Result: draw";
  }
  if (state.winner) {
    return `Winner: ${state.winner}`;
  }
  return `Current player: ${state.current_player}`;
}

async function handleCellClick(event) {
  const target = event.currentTarget;
  const row = Number(target.dataset.row);
  const col = Number(target.dataset.col);
  await postAction("/move", { row, col });
}

async function postAction(url, body = {}) {
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Request failed");
    }
    currentState = data;
    renderState(data);
    clearMessage();
  } catch (error) {
    showMessage(error.message, true);
  }
}

function clearMessage() {
  messageElement.textContent = "";
  messageElement.classList.remove("error");
}

function showMessage(text, isError = false) {
  messageElement.textContent = text;
  messageElement.classList.toggle("error", isError);
}

undoButton.addEventListener("click", () => postAction("/undo"));
resetButton.addEventListener("click", () => {
  const size = Number(sizeInput.value) || currentState?.size || 15;
  const aiSide = aiSelect.value;
  postAction("/reset", { size, ai_side: aiSide });
});
aiSelect.addEventListener("change", () => {
  const aiSide = aiSelect.value;
  postAction("/config", { ai_side: aiSide });
});

document.addEventListener("DOMContentLoaded", () => {
  fetchState();
});
