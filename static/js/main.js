document.addEventListener('DOMContentLoaded', () => {
    // Platform selection (hidden input logic)
    const platformBtns = document.querySelectorAll('.platform-btn');
    const platformInput = document.getElementById('platform-input');
    
    platformBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            platformBtns.forEach(b => {
                b.classList.remove('active', 'border-primary-container/60', 'bg-primary-container/10', 'text-primary-container');
                b.classList.add('border-white/5', 'bg-white/5', 'text-on-surface-variant');
            });
            
            // Add active styles to clicked button
            btn.classList.add('active', 'border-primary-container/60', 'bg-primary-container/10', 'text-primary-container');
            btn.classList.remove('border-white/5', 'bg-white/5', 'text-on-surface-variant');
            
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
                
                // Also trigger click on matching platform button to sync UI state
                const targetBtn = document.querySelector(`.platform-btn[data-platform="${plat}"]`);
                if (targetBtn) {
                    targetBtn.click();
                }
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
                submitBtn.style.opacity = '0.8';
                const labelText = submitBtn.querySelector('span');
                if (labelText) labelText.textContent = 'SCRANNING HANDLE...';
            }
        });
    }
    
    // Threat Score Progress Bar Animation (uses data-score attribute)
    const threatBar = document.querySelector('.threat-progress-bar-fill');
    if (threatBar) {
        const score = threatBar.getAttribute('data-score') || '0%';
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
            reportModal.classList.remove('hidden');
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

    // Terminal widget live log simulation & tab switching
    const terminalContent = document.getElementById('console-tab-terminal');
    if (terminalContent) {
        const logs = [
            { text: "[INFO] Initializing SECURELOCK Core Engine v4.2...", type: "info" },
            { text: "[LOAD] Loading Random Forest High-Recall Ensemble voting node...", type: "info" },
            { text: "[LOAD] Loading K-Nearest Neighbors 2D spatial anomaly mapper...", type: "info" },
            { text: "[SCAN] Social integrity network thread bypass ready.", type: "success" },
            { text: "[ALERT] Direct scraping redirects detected on Twitter. Switching to lite DDG cache search bypass.", type: "warning" },
            { text: "[INFO] Standby for target handles query...", type: "info" },
            { text: "[ALERT] Impersonation anomaly caught in database matching: ID 895737", type: "error" },
            { text: "[ACTION] Logging metrics signature to SQLITE logs...", type: "success" },
            { text: "[SCAN] Instagram Googlebot query bypass: Active.", type: "info" },
            { text: "[SCAN] Facebook scraping cookie and language tunnels: Active.", type: "success" },
            { text: "[SCAN] LinkedIn HTML line-by-line body scanner: Ready.", type: "success" },
            { text: "[SUCCESS] Integrity defense shielding active on all ports.", type: "success" }
        ];

        let index = 0;
        function addLog() {
            const log = logs[index % logs.length];
            const div = document.createElement('div');
            div.className = "flex gap-2 py-0.5 leading-relaxed font-mono";
            
            let colorClass = "text-on-surface-variant/70";
            if (log.type === "error") colorClass = "text-error";
            if (log.type === "warning") colorClass = "text-secondary-fixed-dim";
            if (log.type === "success") colorClass = "text-primary-fixed";

            const timestamp = new Date().toISOString().slice(11, 19) + "." + Math.floor(Math.random() * 1000).toString().padStart(3, '0');
            div.innerHTML = `<span class="text-white/20 select-none">${timestamp}</span><span class="${colorClass}">${log.text}</span>`;
            
            terminalContent.appendChild(div);
            terminalContent.scrollTop = terminalContent.scrollHeight;
            
            // Keep terminal clean, trim oldest logs
            if (terminalContent.children.length > 50) {
                terminalContent.removeChild(terminalContent.firstChild);
            }
            
            index++;
            setTimeout(addLog, Math.random() * 1800 + 400);
        }
        addLog();
    }

    // Global Console Tab Switching Handler
    window.switchConsoleTab = function(tabName) {
        const tabs = ['terminal', 'audit', 'shield', 'intelligence'];
        
        // Update tab visibility
        tabs.forEach(tab => {
            const el = document.getElementById(`console-tab-${tab}`);
            const btn = document.getElementById(`btn-console-${tab}`);
            if (el && btn) {
                if (tab === tabName) {
                    el.classList.remove('hidden');
                    // Add active button classes
                    btn.className = "px-2 py-1 rounded bg-primary-fixed/15 text-primary-fixed font-bold border border-primary-fixed/25 transition-all uppercase tracking-wider flex items-center gap-1";
                } else {
                    el.classList.add('hidden');
                    // Add inactive button classes
                    btn.className = "px-2 py-1 rounded text-on-surface-variant/75 hover:bg-white/5 transition-all uppercase tracking-wider flex items-center gap-1";
                }
            }
        });
    };
});
