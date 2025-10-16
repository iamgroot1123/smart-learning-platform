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

            if (userAnswer === '') {
                feedbackP.textContent = 'Please enter an answer.';
                feedbackP.style.color = 'red';
                feedbackP.style.display = 'block';
            } else {
                // For now, just acknowledge submission (could add AI grading later)
                feedbackP.textContent = 'Answer submitted! (Grading feature can be added later)';
                feedbackP.style.color = 'green';
                feedbackP.style.display = 'block';
                textarea.disabled = true;
                e.target.disabled = true;
            }
        }
    });

    // Form submission via AJAX
    const form = document.getElementById('upload-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            console.log('Form submitted via AJAX');

            const formData = new FormData(form);

            fetch('/generate', {
                method: 'POST',
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                // Store data in state
                chunks = data.chunks || [];
                questions = data.questions || [];

                // Render results in the quiz tab
                renderResults();

                // Switch to quiz tab
                document.querySelector('[data-tab="quiz"]').click();
            })
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while generating questions.');
            });
        });
    }
});

// Function to render chunks and questions
function renderResults() {
    // Render chunks
    const chunksContainer = document.getElementById('chunks-container');
    chunksContainer.innerHTML = '';
    if (chunks.length > 0) {
        chunksContainer.innerHTML = '<h2>📖 Sample Chunks</h2><ul>' +
            chunks.slice(0, 5).map((chunk, index) =>
                `<li><strong>Chunk ${index + 1}:</strong> ${chunk.substring(0, 300)}...</li>`
            ).join('') +
            '</ul>';
    } else {
        chunksContainer.innerHTML = '<p>No chunks available.</p>';
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
                <button class="submit-answer-btn">Submit Answer</button>
                <p class="feedback" style="display: none;"></p>
            </div>
        `).join('');
    } else {
        shortContainer.innerHTML = '<p>No Short Answer questions generated.</p>';
    }
}
