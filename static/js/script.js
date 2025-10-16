// Add any JavaScript functionality here if needed
console.log("Smart Learning Platform loaded");

// Example: Add form validation or dynamic elements
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(e) {
            // Add any pre-submit logic here
            console.log('Form submitted');
        });
    }
});
