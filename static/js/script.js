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
    const container = document.getElementById('results-container');
    container.innerHTML = '';

    // Render chunks
    if (chunks.length > 0) {
        const chunksSection = document.createElement('div');
        chunksSection.innerHTML = '<h2>📖 Sample Chunks</h2><ul>' +
            chunks.slice(0, 5).map((chunk, index) =>
                `<li><strong>Chunk ${index + 1}:</strong> ${chunk.substring(0, 300)}...</li>`
            ).join('') +
            '</ul>';
        container.appendChild(chunksSection);
    }

    // Render questions
    if (questions.length > 0) {
        const questionsSection = document.createElement('div');
        questionsSection.className = 'questions';
        questionsSection.innerHTML = '<h2>Questions</h2>' +
            questions.map(q => {
                if (q.type === 'mcq') {
                    return `
                        <div class="question">
                            <h3>MCQ ${q.id}: ${q.question}</h3>
                            <ul>
                                ${q.options.map(option => `<li>${option}</li>`).join('')}
                            </ul>
                            <p><strong>Answer:</strong> ${q.answer}</p>
                        </div>
                    `;
                } else {
                    return `
                        <div class="question">
                            <h3>Short Q ${q.id}: ${q.question}</h3>
                        </div>
                    `;
                }
            }).join('');
        container.appendChild(questionsSection);
    }

    if (chunks.length === 0 && questions.length === 0) {
        container.innerHTML = '<p>No results to display. Please upload files and generate questions.</p>';
    }
}
