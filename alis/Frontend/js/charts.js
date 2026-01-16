/**
 * ALIS Charts Module
 * Chart.js based visualizations
 */

const Charts = {
    // Chart instances
    instances: {},

    // Common chart options
    defaultOptions: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                labels: {
                    color: '#94a3b8',
                    font: { family: 'Inter' }
                }
            }
        },
        scales: {
            x: {
                ticks: { color: '#64748b' },
                grid: { color: 'rgba(255,255,255,0.05)' }
            },
            y: {
                ticks: { color: '#64748b' },
                grid: { color: 'rgba(255,255,255,0.05)' }
            }
        }
    },

    // Colors
    colors: {
        primary: '#6366f1',
        success: '#10b981',
        warning: '#f59e0b',
        danger: '#ef4444',
        info: '#3b82f6',
        critical: '#dc2626',
        high: '#f97316',
        medium: '#eab308',
        low: '#22c55e'
    },

    /**
     * Initialize all dashboard charts
     */
    init() {
        this.initRiskChart();
        this.initTrendChart();
    },

    /**
     * Destroy a chart instance
     */
    destroy(chartId) {
        if (this.instances[chartId]) {
            this.instances[chartId].destroy();
            delete this.instances[chartId];
        }
    },

    /**
     * Risk distribution doughnut chart
     */
    initRiskChart() {
        const ctx = document.getElementById('risk-chart');
        if (!ctx) return;

        this.destroy('risk');

        this.instances.risk = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Critical', 'High', 'Medium', 'Low'],
                datasets: [{
                    data: [0, 0, 0, 0],
                    backgroundColor: [
                        this.colors.critical,
                        this.colors.high,
                        this.colors.medium,
                        this.colors.low
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#94a3b8',
                            font: { family: 'Inter' },
                            padding: 20,
                            usePointStyle: true
                        }
                    }
                }
            }
        });
    },

    /**
     * Update risk chart with new data
     */
    updateRiskChart(data) {
        if (!this.instances.risk) return;

        this.instances.risk.data.datasets[0].data = [
            data.critical_count || 0,
            data.high_count || 0,
            data.medium_count || 0,
            data.low_count || 0
        ];
        this.instances.risk.update();
    },

    /**
     * Trend line chart
     */
    initTrendChart() {
        const ctx = document.getElementById('trend-chart');
        if (!ctx) return;

        this.destroy('trend');

        this.instances.trend = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Bio Updates',
                        data: [],
                        borderColor: this.colors.primary,
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'Demo Updates',
                        data: [],
                        borderColor: this.colors.success,
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        fill: true,
                        tension: 0.4
                    },
                    {
                        label: 'Mobile Updates',
                        data: [],
                        borderColor: this.colors.warning,
                        backgroundColor: 'rgba(245, 158, 11, 0.1)',
                        fill: true,
                        tension: 0.4
                    }
                ]
            },
            options: {
                ...this.defaultOptions,
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#94a3b8',
                            font: { family: 'Inter' },
                            usePointStyle: true
                        }
                    }
                }
            }
        });
    },

    /**
     * Update trend chart with time series data
     */
    updateTrendChart(data) {
        if (!this.instances.trend || !data.data) return;

        const labels = data.data.map(d => {
            const date = new Date(d.date);
            return date.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' });
        });

        this.instances.trend.data.labels = labels;
        this.instances.trend.data.datasets[0].data = data.data.map(d => d.bio_updates);
        this.instances.trend.data.datasets[1].data = data.data.map(d => d.demo_updates);
        this.instances.trend.data.datasets[2].data = data.data.map(d => d.mobile_updates);
        this.instances.trend.update();
    },

    /**
     * State comparison bar chart
     */
    initStateChart(canvasId = 'state-chart') {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        this.destroy('state');

        this.instances.state = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Critical Pincodes',
                    data: [],
                    backgroundColor: this.colors.critical
                }, {
                    label: 'High Risk',
                    data: [],
                    backgroundColor: this.colors.high
                }]
            },
            options: {
                ...this.defaultOptions,
                indexAxis: 'y',
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#94a3b8',
                            font: { family: 'Inter' }
                        }
                    }
                }
            }
        });
    },

    updateStateChart(states) {
        if (!this.instances.state) return;

        // Sort by critical count
        const sorted = [...states].sort((a, b) => b.critical_pincodes - a.critical_pincodes).slice(0, 10);

        this.instances.state.data.labels = sorted.map(s => s.state);
        this.instances.state.data.datasets[0].data = sorted.map(s => s.critical_pincodes);
        this.instances.state.data.datasets[1].data = sorted.map(s => s.high_risk_pincodes);
        this.instances.state.update();
    },

    /**
     * Cluster distribution chart
     */
    initClusterChart(canvasId = 'cluster-chart') {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        this.destroy('cluster');

        this.instances.cluster = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        '#6366f1', '#10b981', '#f59e0b',
                        '#ef4444', '#3b82f6', '#8b5cf6',
                        '#ec4899', '#14b8a6'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            color: '#94a3b8',
                            font: { family: 'Inter', size: 11 }
                        }
                    }
                }
            }
        });
    },

    updateClusterChart(clusters) {
        if (!this.instances.cluster) return;

        this.instances.cluster.data.labels = clusters.map(c => c.name || `Cluster ${c.id}`);
        this.instances.cluster.data.datasets[0].data = clusters.map(c => c.pincode_count);
        this.instances.cluster.update();
    },

    /**
     * Forecast line chart
     */
    initForecastChart(canvasId = 'forecast-chart') {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;

        this.destroy('forecast');

        this.instances.forecast = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Predicted',
                    data: [],
                    borderColor: this.colors.primary,
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    fill: false,
                    tension: 0.4
                }, {
                    label: 'Upper Bound',
                    data: [],
                    borderColor: 'rgba(99, 102, 241, 0.3)',
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                }, {
                    label: 'Lower Bound',
                    data: [],
                    borderColor: 'rgba(99, 102, 241, 0.3)',
                    borderDash: [5, 5],
                    fill: '-1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    pointRadius: 0
                }]
            },
            options: {
                ...this.defaultOptions,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#94a3b8',
                            font: { family: 'Inter' }
                        }
                    }
                }
            }
        });
    },

    updateForecastChart(forecasts) {
        if (!this.instances.forecast) return;

        const labels = forecasts.map(f => {
            const date = new Date(f.prediction_date);
            return date.toLocaleDateString('en-IN', { month: 'short', day: 'numeric' });
        });

        this.instances.forecast.data.labels = labels;
        this.instances.forecast.data.datasets[0].data = forecasts.map(f => f.predicted_value);
        this.instances.forecast.data.datasets[1].data = forecasts.map(f => f.upper_bound);
        this.instances.forecast.data.datasets[2].data = forecasts.map(f => f.lower_bound);
        this.instances.forecast.update();
    }
};

// Export
window.Charts = Charts;
