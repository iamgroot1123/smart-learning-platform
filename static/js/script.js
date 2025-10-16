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
            <div class="question">
                <h3>MCQ ${q.id}: ${q.question}</h3>
                <ul>
                    ${q.options.map(option => `<li>${option}</li>`).join('')}
                </ul>
                <p><strong>Answer:</strong> ${q.answer}</p>
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
            <div class="question">
                <h3>Short Q ${q.id}: ${q.question}</h3>
            </div>
        `).join('');
    } else {
        shortContainer.innerHTML = '<p>No Short Answer questions generated.</p>';
    }
}
