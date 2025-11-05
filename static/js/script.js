// Smart Learning Platform SPA Script
console.log("Smart Learning Platform loaded");

// State management
let chunks = [];
let questions = [];

// Tab switching functionality
document.addEventListener('DOMContentLoaded', function() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');

            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked button and corresponding content
            this.classList.add('active');
            document.getElementById(tabName + '-tab').classList.add('active');
        });
    });

    // Sub-tab switching functionality
    const subTabButtons = document.querySelectorAll('.sub-tab-button');
    const subTabContents = document.querySelectorAll('.sub-tab-content');

    subTabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const subTabName = this.getAttribute('data-subtab');

            // Remove active class from all sub-buttons and sub-contents
            subTabButtons.forEach(btn => btn.classList.remove('active'));
            subTabContents.forEach(content => content.classList.remove('active'));

            // Add active class to clicked sub-button and corresponding sub-content
            this.classList.add('active');
            document.getElementById(subTabName + '-subtab').classList.add('active');
        });
    });

    // Event delegation for dynamic elements (reveal answer buttons and MCQ submit)
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('reveal-btn')) {
            const answerP = e.target.nextElementSibling;
            answerP.style.display = 'block';
            e.target.style.display = 'none';
        }

        if (e.target.classList.contains('submit-mcq-btn')) {
            const questionDiv = e.target.closest('.mcq-question');
            const form = questionDiv.querySelector('.mcq-form');
            const selectedOption = form.querySelector('input[name^="mcq-"]:checked');
            const evaluationDiv = questionDiv.querySelector('.evaluation');
            const correctAnswer = e.target.getAttribute('data-answer');
            const options = JSON.parse(e.target.getAttribute('data-options'));

            if (!selectedOption) {
                evaluationDiv.innerHTML = '<p style="color: red;">Please select an answer before submitting.</p>';
                evaluationDiv.style.display = 'block';
                return;
            }

            const userAnswer = selectedOption.value;
            const isCorrect = userAnswer === correctAnswer;

            // Disable form and button
            form.querySelectorAll('input').forEach(input => input.disabled = true);
            e.target.disabled = true;

            // Display all options with labels
            const optionsHtml = options.map((option, index) => {
                const label = String.fromCharCode(65 + index);
                let style = '';
                if (option === correctAnswer) {
                    style = 'color: green; font-weight: bold;';
                } else if (option === userAnswer && !isCorrect) {
                    style = 'color: red;';
                }
                return `<p style="${style}">${label}. ${option}</p>`;
            }).join('');

            if (isCorrect) {
                evaluationDiv.innerHTML = `
                    <p style="color: green; font-weight: bold;">Correct!</p>
                    ${optionsHtml}
                `;
            } else {
                evaluationDiv.innerHTML = `
                    <p style="color: red; font-weight: bold;">Wrong!</p>
                    ${optionsHtml}
                `;
            }
            evaluationDiv.style.display = 'block';
        }

        if (e.target.classList.contains('submit-answer-btn')) {
            const questionDiv = e.target.closest('.short-question');
            const textarea = questionDiv.querySelector('.answer-input');
            const feedbackP = questionDiv.querySelector('.feedback');
            const userAnswer = textarea.value.trim();
            const correctAnswer = e.target.getAttribute('data-correct-answer');
            const questionId = e.target.getAttribute('data-question-id');

            if (userAnswer === '') {
                feedbackP.textContent = 'Please enter an answer.';
                feedbackP.style.color = 'red';
                feedbackP.style.display = 'block';
            } else {
                // Disable inputs
                textarea.disabled = true;
                e.target.disabled = true;

                // Send to backend for evaluation
                fetch('/evaluate_short', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        user_answer: userAnswer,
                        correct_answer: correctAnswer,
                        question_id: questionId
                    })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        feedbackP.textContent = 'Error evaluating answer.';
                        feedbackP.style.color = 'red';
                    } else {
                        feedbackP.innerHTML = `
                            <strong>Your answer:</strong> "${userAnswer}"<br>
                            <strong>Correct answer:</strong> "${data.correct_answer}"<br>
                            <strong>Similarity:</strong> ${data.similarity}%<br>
                            <strong>Status:</strong> <span style="color: ${data.color};">${data.status}</span>
                        `;
                        feedbackP.style.display = 'block';
                    }
                })
                .catch(error => {
                    console.error('Error:', error);
                    feedbackP.textContent = 'Error evaluating answer.';
                    feedbackP.style.color = 'red';
                    feedbackP.style.display = 'block';
                });
            }
        }
    });

    // Form submission via AJAX with progress and overlay
    const form = document.getElementById('upload-form');
    const overlay = document.getElementById('loading-overlay');
    const loadingMessage = document.getElementById('loading-message');
    const progressBar = document.getElementById('upload-progress');
    const progressFill = document.getElementById('upload-progress-fill');

    function showOverlay(message) {
        if (loadingMessage) loadingMessage.textContent = message || 'Working…';
        if (overlay) overlay.style.display = 'flex';
    }
    function updateProgress(percent) {
        if (progressBar && progressFill) {
            progressBar.style.display = 'block';
            progressFill.style.width = Math.min(100, Math.max(0, percent)) + '%';
        }
    }
    function hideOverlay() {
        if (overlay) overlay.style.display = 'none';
        if (progressBar && progressFill) {
            progressBar.style.display = 'none';
            progressFill.style.width = '0%';
        }
    }

    function showToast(message, type = 'info', timeout = 4000) {
        const container = document.getElementById('toast-container');
        if (!container) return;
        const t = document.createElement('div');
        t.className = 'toast ' + (type || 'info');
        t.textContent = message;
        container.appendChild(t);
        setTimeout(() => { t.style.opacity = '0'; }, timeout - 500);
        setTimeout(() => { t.remove(); }, timeout);
    }

    function disableForm() {
        form.querySelectorAll('input, button, textarea').forEach(el => el.disabled = true);
    }
    function enableForm() {
        form.querySelectorAll('input, button, textarea').forEach(el => el.disabled = false);
    }

    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();

            const formData = new FormData(form);

            // Use XHR to get upload progress events
            const xhr = new XMLHttpRequest();
            xhr.open('POST', '/generate');

            showOverlay('Uploading files...');
            disableForm();

            xhr.upload.addEventListener('progress', function(ev) {
                if (ev.lengthComputable) {
                    const percent = Math.round((ev.loaded / ev.total) * 100);
                    updateProgress(percent);
                    loadingMessage.textContent = `Uploading files (${percent}%)`;
                }
            });

            xhr.addEventListener('load', function() {
                // Upload finished; show indeterminate processing
                updateProgress(100);
                loadingMessage.textContent = 'Generating questions — this may take a moment.';
            });

            xhr.addEventListener('readystatechange', function() {
                if (xhr.readyState === 4) {
                    try {
                        const data = JSON.parse(xhr.responseText);
                        if (data.error) {
                            showToast('Error: ' + data.error, 'error');
                        } else {
                            chunks = data.chunks || [];
                            questions = data.questions || [];
                            renderResults();
                            document.querySelector('[data-tab="quiz"]').click();
                            showToast('Questions generated successfully', 'success');
                        }
                    } catch (err) {
                        console.error('Invalid JSON response', err);
                        showToast('Unexpected server response', 'error');
                    }
                    hideOverlay();
                    enableForm();
                }
            });

            xhr.addEventListener('error', function() {
                hideOverlay();
                enableForm();
                showToast('Network error during upload', 'error');
            });

            xhr.send(formData);
        });
    }
});

// Function to render chunks and questions
function renderResults() {
    // Do not display sample chunks in the quiz view (user requested)
    const chunksContainer = document.getElementById('chunks-container');
    if (chunksContainer) {
        chunksContainer.innerHTML = '';
    }

    // Separate MCQs and Short Answers
    const mcqs = questions.filter(q => q.type === 'mcq');
    const shorts = questions.filter(q => q.type === 'short');

    // Render MCQs
    const mcqContainer = document.getElementById('mcq-container');
    mcqContainer.innerHTML = '';
    if (mcqs.length > 0) {
        mcqContainer.innerHTML = mcqs.map(q => `
            <div class="question mcq-question">
                <h3>MCQ ${q.id}: ${q.question}</h3>
                <form class="mcq-form">
                    ${q.options.map((option, index) => `
                        <label class="option">
                            <input type="radio" name="mcq-${q.id}" value="${option}">
                            <span class="option-label">${String.fromCharCode(65 + index)}.</span> ${option}
                        </label>
                    `).join('')}
                </form>
                <button class="submit-mcq-btn" data-answer="${q.answer}" data-options='${JSON.stringify(q.options)}'>Submit Answer</button>
                <div class="evaluation" style="display: none;"></div>
            </div>
        `).join('');
    } else {
        mcqContainer.innerHTML = '<p>No MCQs generated.</p>';
    }

    // Render Short Answers
    const shortContainer = document.getElementById('short-container');
    shortContainer.innerHTML = '';
    if (shorts.length > 0) {
        shortContainer.innerHTML = shorts.map(q => `
            <div class="question short-question">
                <h3>Short Q ${q.id}: ${q.question}</h3>
                <textarea class="answer-input" placeholder="Enter your answer here..."></textarea>
                <button class="submit-answer-btn" data-correct-answer="${q.correct_answer}" data-question-id="${q.id}">Submit Answer</button>
                <p class="feedback" style="display: none;"></p>
            </div>
        `).join('');
    } else {
        shortContainer.innerHTML = '<p>No Short Answer questions generated.</p>';
    }
}
