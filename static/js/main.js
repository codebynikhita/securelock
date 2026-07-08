document.addEventListener('DOMContentLoaded', () => {
    // Platform selection (hidden input logic)
    const platformBtns = document.querySelectorAll('.platform-btn');
    const platformInput = document.getElementById('platform-input');
    
    platformBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            platformBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (platformInput) {
                platformInput.value = btn.getAttribute('data-platform');
            }
        });
    });
    
    // Suggestion chips
    const chips = document.querySelectorAll('.suggestion-chip');
    const searchInput = document.getElementById('search-input');
    const searchForm = document.getElementById('search-form');
    
    chips.forEach(chip => {
        chip.addEventListener('click', () => {
            if (searchInput) {
                searchInput.value = chip.getAttribute('data-username');
            }
            const plat = chip.getAttribute('data-platform');
            if (plat && platformInput) {
                platformInput.value = plat;
            }
            if (searchForm) {
                searchForm.dispatchEvent(new Event('submit'));
                searchForm.submit();
            }
        });
    });
    
    // Form submission indicator
    if (searchForm) {
        searchForm.addEventListener('submit', () => {
            const spinIcon = document.getElementById('spin-icon');
            if (spinIcon) {
                spinIcon.style.display = 'inline-block';
            }
            const submitBtn = searchForm.querySelector('.btn-initialize');
            if (submitBtn) {
                submitBtn.style.opacity = '0.85';
                submitBtn.querySelector('span').textContent = 'SCANNING PROFILE...';
            }
        });
    }
    
    // Threat Score Progress Bar Animation
    const threatBar = document.querySelector('.threat-progress-bar-fill');
    if (threatBar) {
        const score = threatBar.parentElement.previousElementSibling.querySelector('.threat-value').textContent;
        // Set width dynamically to trigger transition animation
        threatBar.style.width = '0%';
        setTimeout(() => {
            threatBar.style.width = score;
        }, 150);
    }
    
    // Report Modal
    const btnReport = document.getElementById('btn-report');
    const reportModal = document.getElementById('report-modal');
    const btnCancelReport = document.getElementById('btn-cancel-report');
    const reportForm = document.getElementById('report-form');
    
    if (btnReport && reportModal) {
        btnReport.addEventListener('click', () => {
            reportModal.style.display = 'flex';
        });
    }
    
    if (btnCancelReport && reportModal) {
        btnCancelReport.addEventListener('click', () => {
            reportModal.style.display = 'none';
        });
    }
    
    window.addEventListener('click', (e) => {
        if (e.target === reportModal) {
            reportModal.style.display = 'none';
        }
    });
    
    if (reportForm) {
        reportForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const username = document.getElementById('report-username').value;
            const platform = document.getElementById('report-platform').value;
            const score = document.getElementById('report-score').value;
            const reason = document.getElementById('report-reason').value;
            
            const formData = new FormData();
            formData.append('username', username);
            formData.append('platform', platform);
            formData.append('risk_score', score);
            formData.append('reason', reason);
            
            fetch('/report', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                if (data.status === 'success') {
                    alert(data.message);
                    reportModal.style.display = 'none';
                    reportForm.reset();
                } else {
                    alert('Error: ' + data.message);
                }
            })
            .catch(err => {
                console.error(err);
                alert('An error occurred. Please try again.');
            });
        });
    }
});
