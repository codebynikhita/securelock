// Admin dashboard statistics rendering
document.addEventListener('DOMContentLoaded', () => {
    // 1. Classification distribution Pie Chart
    const classChartCanvas = document.getElementById('class-dist-chart');
    if (classChartCanvas) {
        const genuine = parseInt(classChartCanvas.getAttribute('data-genuine') || 0);
        const suspicious = parseInt(classChartCanvas.getAttribute('data-suspicious') || 0);
        const fake = parseInt(classChartCanvas.getAttribute('data-fake') || 0);
        
        new Chart(classChartCanvas, {
            type: 'doughnut',
            data: {
                labels: ['Genuine', 'Suspicious', 'Fake'],
                datasets: [{
                    data: [genuine, suspicious, fake],
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.7)',  // Success green
                        'rgba(245, 158, 11, 0.7)',  // Warning orange
                        'rgba(239, 68, 68, 0.7)'    // Danger red
                    ],
                    borderColor: [
                        '#10b981',
                        '#f59e0b',
                        '#ef4444'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#9ca3af',
                            font: {
                                family: "'Outfit', sans-serif",
                                size: 12
                            }
                        }
                    }
                },
                cutout: '65%'
            }
        });
    }

    // 2. KNN Distance Visualization Chart
    const knnChartCanvas = document.getElementById('knn-distance-chart');
    if (knnChartCanvas) {
        const queryUsername = knnChartCanvas.getAttribute('data-query') || '';
        const queryPlatform = knnChartCanvas.getAttribute('data-platform') || 'twitter';
        
        // Fetch visualization data from Flask
        let url = `/admin/visualize`;
        if (queryUsername) {
            url += `?query_username=${encodeURIComponent(queryUsername)}&query_platform=${encodeURIComponent(queryPlatform)}`;
        }
        
        fetch(url)
            .then(res => res.json())
            .then(data => {
                if (data.error) {
                    knnChartCanvas.parentNode.innerHTML = `<div style="color: #6b7280; font-size: 0.9rem;">${data.error}</div>`;
                    return;
                }
                
                const datasetPoints = data.dataset_points || [];
                const queryPoint = data.query_point;
                const nearestNeighbors = data.nearest_neighbors || [];
                
                // Group points by label
                const genuinePoints = datasetPoints.filter(p => p.label === 'Genuine').map(p => ({ x: p.x, y: p.y, label: p.username }));
                const fakePoints = datasetPoints.filter(p => p.label === 'Fake').map(p => ({ x: p.x, y: p.y, label: p.username }));
                const clonePoints = datasetPoints.filter(p => p.label === 'Clone').map(p => ({ x: p.x, y: p.y, label: p.username }));
                
                const datasets = [
                    {
                        label: 'Genuine Reference Group',
                        data: genuinePoints,
                        backgroundColor: 'rgba(16, 185, 129, 0.4)',
                        borderColor: '#10b981',
                        pointRadius: 5,
                        pointHoverRadius: 7
                    },
                    {
                        label: 'Fake Reference Group',
                        data: fakePoints,
                        backgroundColor: 'rgba(239, 68, 68, 0.4)',
                        borderColor: '#ef4444',
                        pointRadius: 5,
                        pointHoverRadius: 7
                    },
                    {
                        label: 'Clone Reference Group',
                        data: clonePoints,
                        backgroundColor: 'rgba(168, 85, 247, 0.4)',
                        borderColor: '#a855f7',
                        pointRadius: 5,
                        pointHoverRadius: 7
                    }
                ];
                
                // Add query point and connections if it exists
                if (queryPoint) {
                    datasets.push({
                        label: `Queried: @${queryUsername}`,
                        data: [{ x: queryPoint.x, y: queryPoint.y, label: queryUsername }],
                        backgroundColor: '#60a5fa',
                        borderColor: '#3b82f6',
                        pointRadius: 9,
                        pointHoverRadius: 11,
                        borderWidth: 3,
                        showLine: false
                    });
                }
                
                // Build Chart
                const chartInstance = new Chart(knnChartCanvas, {
                    type: 'scatter',
                    data: {
                        datasets: datasets
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'top',
                                labels: {
                                    color: '#9ca3af',
                                    font: { family: "'Outfit', sans-serif", size: 10 }
                                }
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        const point = context.raw;
                                        return `@${point.label} (Ratio Log: ${point.x.toFixed(2)}, Similarity: ${point.y.toFixed(2)})`;
                                    }
                                }
                            }
                        },
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: 'Follower / Following Ratio (Log Scale)',
                                    color: '#6b7280',
                                    font: { family: "'Outfit', sans-serif" }
                                },
                                grid: { color: 'rgba(255,255,255,0.03)' },
                                ticks: { color: '#6b7280' }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Content Similarity Score (0 - 1)',
                                    color: '#6b7280',
                                    font: { family: "'Outfit', sans-serif" }
                                },
                                grid: { color: 'rgba(255,255,255,0.03)' },
                                ticks: { color: '#6b7280' },
                                min: 0,
                                max: 1
                            }
                        }
                    },
                    plugins: [{
                        id: 'knnLines',
                        beforeDraw: function(chart) {
                            if (!queryPoint || nearestNeighbors.length === 0) return;
                            
                            const ctx = chart.ctx;
                            const xAxis = chart.scales.x;
                            const yAxis = chart.scales.y;
                            
                            // Find the query point pixel position
                            const qPixelX = xAxis.getPixelForValue(queryPoint.x);
                            const qPixelY = yAxis.getPixelForValue(queryPoint.y);
                            
                            // Draw lines to nearest neighbors
                            ctx.save();
                            ctx.beginPath();
                            ctx.strokeStyle = 'rgba(96, 165, 250, 0.4)';
                            ctx.lineWidth = 1.5;
                            ctx.setLineDash([4, 4]);
                            
                            nearestNeighbors.forEach(n => {
                                const nPixelX = xAxis.getPixelForValue(n.x);
                                const nPixelY = yAxis.getPixelForValue(n.y);
                                ctx.moveTo(qPixelX, qPixelY);
                                ctx.lineTo(nPixelX, nPixelY);
                            });
                            
                            ctx.stroke();
                            ctx.restore();
                        }
                    }]
                });
            })
            .catch(err => {
                console.error(err);
                knnChartCanvas.parentNode.innerHTML = `<div style="color: #6b7280; font-size: 0.9rem;">Failed to load visualization data.</div>`;
            });
    }
});
