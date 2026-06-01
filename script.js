// ============================================================
// Sudoku Web App
// ============================================================

// ------------------------------------------------------------
// 상수
// ------------------------------------------------------------
const BOARD_SIZE = 9;
const BOX_SIZE = 3;
const TOTAL_CELLS = BOARD_SIZE * BOARD_SIZE;
const MAX_MISTAKES = 3;

const DIFFICULTY = {
    easy:   { min: 35, max: 40 },
    medium: { min: 30, max: 34 },
    hard:   { min: 25, max: 29 }
};

const STATUS = {
    PLAYING: 'playing',
    WON: 'won',
    LOST: 'lost'
};

// ------------------------------------------------------------
// 게임 상태
// ------------------------------------------------------------
const State = {
    solution: null,
    puzzle: null,
    given: null,
    userInput: null,
    errors: null,
    selected: null,         // { row, col } | null
    difficulty: 'medium',
    mistakes: 0,
    elapsed: 0,             // 경과 초
    timerId: null,
    status: STATUS.PLAYING
};

// ------------------------------------------------------------
// DOM 캐시
// ------------------------------------------------------------
const DOM = {};

function cacheDOM() {
    DOM.board = document.getElementById('board');
    DOM.timer = document.getElementById('timer');
    DOM.mistakes = document.getElementById('mistakes');
    DOM.difficulty = document.getElementById('difficulty');
    DOM.newGameBtn = document.getElementById('new-game');
    DOM.keypad = document.querySelector('.keypad');
    DOM.modal = document.getElementById('modal');
    DOM.modalTitle = document.getElementById('modal-title');
    DOM.modalMessage = document.getElementById('modal-message');
    DOM.modalClose = document.getElementById('modal-close');
}

// ------------------------------------------------------------
// 유틸
// ------------------------------------------------------------
function randomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function shuffle(arr) {
    const a = arr.slice();
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

function deepCopy(board) {
    return board.map(row => row.slice());
}

function emptyBoard() {
    return Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(0));
}

function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

function forEachCell(callback) {
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) callback(r, c);
    }
}

// ------------------------------------------------------------
// 스도쿠 규칙 & 보드 생성
// ------------------------------------------------------------
function isValidPlacement(board, row, col, num) {
    for (let i = 0; i < BOARD_SIZE; i++) {
        if (board[row][i] === num) return false;
        if (board[i][col] === num) return false;
    }
    const boxRow = Math.floor(row / BOX_SIZE) * BOX_SIZE;
    const boxCol = Math.floor(col / BOX_SIZE) * BOX_SIZE;
    for (let r = boxRow; r < boxRow + BOX_SIZE; r++) {
        for (let c = boxCol; c < boxCol + BOX_SIZE; c++) {
            if (board[r][c] === num) return false;
        }
    }
    return true;
}

function fillBoard(board) {
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            if (board[r][c] === 0) {
                const nums = shuffle([1, 2, 3, 4, 5, 6, 7, 8, 9]);
                for (const num of nums) {
                    if (isValidPlacement(board, r, c, num)) {
                        board[r][c] = num;
                        if (fillBoard(board)) return true;
                        board[r][c] = 0;
                    }
                }
                return false;
            }
        }
    }
    return true;
}

function generateSolution() {
    const board = emptyBoard();
    fillBoard(board);
    return board;
}

function createPuzzle(solution, difficulty) {
    const range = DIFFICULTY[difficulty] || DIFFICULTY.medium;
    const filledCount = randomInt(range.min, range.max);
    const removeCount = TOTAL_CELLS - filledCount;

    const puzzle = deepCopy(solution);
    const indices = shuffle(Array.from({ length: TOTAL_CELLS }, (_, i) => i));

    for (let i = 0; i < removeCount; i++) {
        const idx = indices[i];
        puzzle[Math.floor(idx / BOARD_SIZE)][idx % BOARD_SIZE] = 0;
    }
    return puzzle;
}

// ------------------------------------------------------------
// 상태 조회
// ------------------------------------------------------------
function valueAt(row, col) {
    return State.given[row][col] ? State.puzzle[row][col] : State.userInput[row][col];
}

function getCellEl(row, col) {
    return DOM.board.children[row * BOARD_SIZE + col];
}

function isWon() {
    for (let r = 0; r < BOARD_SIZE; r++) {
        for (let c = 0; c < BOARD_SIZE; c++) {
            if (valueAt(r, c) !== State.solution[r][c]) return false;
        }
    }
    return true;
}

// ------------------------------------------------------------
// 타이머
// ------------------------------------------------------------
function startTimer() {
    stopTimer();
    State.elapsed = 0;
    renderTimer();
    State.timerId = setInterval(() => {
        State.elapsed++;
        renderTimer();
    }, 1000);
}

function stopTimer() {
    if (State.timerId !== null) {
        clearInterval(State.timerId);
        State.timerId = null;
    }
}

// ------------------------------------------------------------
// 렌더링
// ------------------------------------------------------------
function renderTimer() {
    DOM.timer.textContent = formatTime(State.elapsed);
}

function renderMistakes() {
    DOM.mistakes.textContent = `${State.mistakes}/${MAX_MISTAKES}`;
}

function renderBoard() {
    DOM.board.innerHTML = '';
    forEachCell((r, c) => {
        const cell = document.createElement('div');
        cell.className = 'cell';
        cell.dataset.row = String(r);
        cell.dataset.col = String(c);
        cell.setAttribute('role', 'gridcell');
        DOM.board.appendChild(cell);
        renderCellValue(r, c);
    });
    renderHighlights();
}

function renderCellValue(row, col) {
    const cell = getCellEl(row, col);
    const value = valueAt(row, col);
    cell.textContent = value === 0 ? '' : String(value);

    cell.classList.remove('given', 'user', 'error');
    if (value === 0) return;
    if (State.given[row][col]) cell.classList.add('given');
    else if (State.errors[row][col]) cell.classList.add('error');
    else cell.classList.add('user');
}

function renderHighlights() {
    const sel = State.selected;
    const selValue = sel ? valueAt(sel.row, sel.col) : 0;

    forEachCell((r, c) => {
        const cell = getCellEl(r, c);
        cell.classList.remove('selected', 'same-number');

        if (sel && sel.row === r && sel.col === c) {
            cell.classList.add('selected');
        } else if (selValue !== 0 && valueAt(r, c) === selValue) {
            cell.classList.add('same-number');
        }
    });
}

// ------------------------------------------------------------
// 모달
// ------------------------------------------------------------
function showModal(title, message) {
    DOM.modalTitle.textContent = title;
    DOM.modalMessage.textContent = message;
    DOM.modal.classList.remove('hidden');
}

function hideModal() {
    DOM.modal.classList.add('hidden');
}

// ------------------------------------------------------------
// 게임 흐름
// ------------------------------------------------------------
function startNewGame() {
    hideModal();
    State.difficulty = DOM.difficulty.value;
    State.solution = generateSolution();
    State.puzzle = createPuzzle(State.solution, State.difficulty);
    State.given = State.puzzle.map(row => row.map(v => v !== 0));
    State.userInput = State.puzzle.map(row => row.map(() => 0));
    State.errors = State.puzzle.map(row => row.map(() => false));
    State.selected = null;
    State.mistakes = 0;
    State.status = STATUS.PLAYING;

    renderBoard();
    renderMistakes();
    startTimer();
}

function endGame(result) {
    State.status = result;
    stopTimer();
    if (result === STATUS.WON) {
        showModal('축하합니다!', `완료 시간: ${formatTime(State.elapsed)}`);
    } else {
        showModal('게임 오버', '새로운 게임을 시작하세요.');
    }
}

// ------------------------------------------------------------
// 인터랙션
// ------------------------------------------------------------
function selectCell(row, col) {
    State.selected = { row, col };
    renderHighlights();
}

function inputAt(row, col, num) {
    if (State.status !== STATUS.PLAYING) return;
    if (State.given[row][col]) return;
    if (State.userInput[row][col] === num) return;

    State.userInput[row][col] = num;
    const isError = num !== State.solution[row][col];
    State.errors[row][col] = isError;

    if (isError) {
        State.mistakes++;
        renderMistakes();
    }

    renderCellValue(row, col);
    renderHighlights();

    if (State.mistakes >= MAX_MISTAKES) return endGame(STATUS.LOST);
    if (isWon()) return endGame(STATUS.WON);
}

function eraseAt(row, col) {
    if (State.status !== STATUS.PLAYING) return;
    if (State.given[row][col]) return;
    if (State.userInput[row][col] === 0) return;

    State.userInput[row][col] = 0;
    State.errors[row][col] = false;
    renderCellValue(row, col);
    renderHighlights();
}

// ------------------------------------------------------------
// 이벤트 핸들러
// ------------------------------------------------------------
function handleBoardClick(e) {
    const cell = e.target.closest('.cell');
    if (!cell) return;
    selectCell(parseInt(cell.dataset.row, 10), parseInt(cell.dataset.col, 10));
}

function handleKeydown(e) {
    if (!State.selected) return;
    const { row, col } = State.selected;
    if (e.key >= '1' && e.key <= '9') {
        inputAt(row, col, parseInt(e.key, 10));
        e.preventDefault();
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
        eraseAt(row, col);
        e.preventDefault();
    }
}

function handleKeypadClick(e) {
    const key = e.target.closest('.key');
    if (!key || !State.selected) return;
    const { row, col } = State.selected;
    if (key.dataset.key === 'erase') eraseAt(row, col);
    else inputAt(row, col, parseInt(key.dataset.key, 10));
}

// ------------------------------------------------------------
// 초기화
// ------------------------------------------------------------
function init() {
    cacheDOM();
    DOM.board.addEventListener('click', handleBoardClick);
    DOM.keypad.addEventListener('click', handleKeypadClick);
    DOM.newGameBtn.addEventListener('click', startNewGame);
    DOM.difficulty.addEventListener('change', startNewGame);
    DOM.modalClose.addEventListener('click', startNewGame);
    document.addEventListener('keydown', handleKeydown);
    startNewGame();
}

document.addEventListener('DOMContentLoaded', init);
