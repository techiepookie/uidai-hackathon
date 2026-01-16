/**
 * ALIS Main Application
 * Orchestrates all modules and handles user interactions
 */

const App = {
    currentPage: 'dashboard',
    dashboardData: null,
    states: [],

    /**
     * Initialize the application
     */
    async init() {
        console.log('Initializing ALIS Dashboard...');

        // Setup event listeners
        this.setupEventListeners();

        // Initialize charts
        Charts.init();

        // Load initial data
        await this.loadDashboardData();

        // Check API health
        this.checkHealth();

        console.log('ALIS Dashboard initialized');
    },

    /**
     * Setup all event listeners
     */
    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const page = item.dataset.page;
                this.navigateTo(page);
            });
        });

        // View all links
        document.querySelectorAll('.view-all').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const page = link.dataset.page;
                this.navigateTo(page);
            });
        });

        // Menu toggle (mobile)
        document.getElementById('menu-toggle')?.addEventListener('click', () => {
            document.getElementById('sidebar').classList.toggle('open');
        });

        // Theme toggle
        document.getElementById('theme-toggle')?.addEventListener('click', () => {
            this.toggleTheme();
        });

        // Refresh button
        document.getElementById('refresh-btn')?.addEventListener('click', () => {
            this.refresh();
        });

        // Pincode search
        document.getElementById('pincode-search')?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const pincode = e.target.value.trim();
                if (pincode.length === 6) {
                    this.showPincodeDetail(pincode);
                }
            }
        });

        // Filters
        document.getElementById('state-filter')?.addEventListener('change', (e) => {
            this.filterByState(e.target.value);
        });

        document.getElementById('risk-filter')?.addEventListener('change', (e) => {
            this.filterByRisk(e.target.value);
        });

        document.getElementById('severity-filter')?.addEventListener('change', (e) => {
            this.loadAnomalies();
        });

        document.getElementById('metric-filter')?.addEventListener('change', (e) => {
            this.loadAnomalies();
        });

        // Forecast button
        document.getElementById('forecast-btn')?.addEventListener('click', () => {
            this.generateForecast();
        });

        document.getElementById('forecast-pincode')?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.generateForecast();
            }
        });

        // Modal close
        document.getElementById('modal-close')?.addEventListener('click', () => {
            this.closeModal();
        });

        document.getElementById('pincode-modal')?.addEventListener('click', (e) => {
            if (e.target.id === 'pincode-modal') {
                this.closeModal();
            }
        });

        // Alert close
        document.getElementById('alert-close')?.addEventListener('click', () => {
            document.getElementById('alert-banner').style.display = 'none';
        });
    },

    /**
     * Navigate to a page
     */
    navigateTo(page) {
        // Update navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.toggle('active', item.dataset.page === page);
        });

        // Update page visibility
        document.querySelectorAll('.page').forEach(p => {
            p.style.display = 'none';
        });

        const pageEl = document.getElementById(`page-${page}`);
        if (pageEl) {
            pageEl.style.display = 'block';
        }

        // Update title
        const titles = {
            dashboard: 'Dashboard',
            map: 'India Map View',
            priority: 'Priority Actions',
            analytics: 'Analytics',
            anomalies: 'Anomalies',
            predictions: 'Predictions'
        };
        document.getElementById('page-title').textContent = titles[page] || 'Dashboard';

        // Load page-specific data
        this.currentPage = page;
        this.loadPageData(page);

        // Close mobile sidebar
        document.getElementById('sidebar')?.classList.remove('open');
    },

    /**
     * Load page-specific data
     */
    async loadPageData(page) {
        switch (page) {
            case 'map':
                MapView.init();
                await MapView.loadData();
                break;
            case 'priority':
                await this.loadPriorityData();
                break;
            case 'analytics':
                await this.loadAnalyticsData();
                break;
            case 'anomalies':
                await this.loadAnomalies();
                break;
            case 'predictions':
                Charts.initForecastChart();
                break;
        }
    },

    /**
     * Load dashboard data
     */
    async loadDashboardData() {
        try {
            // Load stats
            const stats = await API.getDashboardStats();
            this.dashboardData = stats;
            this.updateStats(stats);

            // Update risk chart
            Charts.updateRiskChart(stats);

            // Load trend data
            const trend = await API.getTrendAnalysis(30);
            Charts.updateTrendChart(trend);

            // Load priority pincodes
            const priority = await API.getPriorityPincodes(10);
            Tables.renderPriorityTable(priority.pincodes || priority);

            // Load states for filters
            const statesData = await API.getStates();
            this.states = statesData.states || [];
            this.populateStateFilter();

            // Check for anomalies
            const anomalies = await API.getRecentAnomalies(5);
            this.updateAnomalyBadge(anomalies.length);

            if (anomalies.length > 0) {
                this.showAlert(`${anomalies.length} new anomalies detected today`);
            }

        } catch (error) {
            console.error('Failed to load dashboard data:', error);
            this.showAlert('Failed to connect to API. Running in demo mode.');
            this.loadDemoData();
        }
    },

    /**
     * Update stats cards
     */
    updateStats(stats) {
        document.getElementById('stat-critical').textContent = stats.critical_count || 0;
        document.getElementById('stat-high').textContent = stats.high_count || 0;
        document.getElementById('stat-anomalies').textContent = stats.new_anomalies_today || 0;
        document.getElementById('stat-accuracy').textContent =
            `${((stats.model_accuracy || 0) * 100).toFixed(0)}%`;

        // Data freshness
        const hours = stats.data_freshness_hours || 0;
        let freshnessText = hours < 24 ? `${hours}h ago` : `${Math.floor(hours / 24)}d ago`;
        document.getElementById('data-freshness').textContent = freshnessText;
    },

    /**
     * Load priority data
     */
    async loadPriorityData() {
        try {
            const data = await API.getPriorityPincodes(100);
            Tables.renderFullPriorityTable(data.pincodes || data);
        } catch (error) {
            console.error('Failed to load priority data:', error);
        }
    },

    /**
     * Load analytics data
     */
    async loadAnalyticsData() {
        try {
            // Initialize charts if needed
            Charts.initStateChart();
            Charts.initClusterChart();

            // Load state overview
            const states = await API.getStateOverview();
            Charts.updateStateChart(states);

            // Load cluster data
            const clusters = await API.getClusterAnalysis();
            Charts.updateClusterChart(clusters);

        } catch (error) {
            console.error('Failed to load analytics data:', error);
        }
    },

    /**
     * Load anomalies
     */
    async loadAnomalies() {
        try {
            const severity = document.getElementById('severity-filter')?.value;
            const metricType = document.getElementById('metric-filter')?.value;

            const data = await API.getAnomalies({
                days: 7,
                severity: severity || undefined,
                metricType: metricType || undefined,
                limit: 50
            });

            Tables.renderAnomalyCards(data.anomalies || []);

        } catch (error) {
            console.error('Failed to load anomalies:', error);
        }
    },

    /**
     * Generate forecast for a pincode
     */
    async generateForecast() {
        const pincode = document.getElementById('forecast-pincode')?.value.trim();

        if (!pincode || pincode.length !== 6) {
            this.showAlert('Please enter a valid 6-digit pincode');
            return;
        }

        try {
            document.getElementById('forecast-pincode-label').textContent = pincode;
            document.getElementById('forecast-container').style.display = 'block';

            Charts.initForecastChart();

            const data = await API.getPincodeForecast(pincode, 'bio', 30);
            Charts.updateForecastChart(data.forecasts);

        } catch (error) {
            console.error('Failed to generate forecast:', error);
            this.showAlert('Failed to generate forecast. Pincode may not exist.');
        }
    },

    /**
     * Show pincode detail modal
     */
    async showPincodeDetail(pincode) {
        try {
            const detail = await API.getPincodeDetail(pincode);

            document.getElementById('modal-pincode-title').textContent =
                `Pincode: ${pincode}`;
            document.getElementById('modal-body').innerHTML =
                Tables.renderPincodeDetail(pincode, detail);
            document.getElementById('pincode-modal').style.display = 'flex';

        } catch (error) {
            console.error('Failed to load pincode detail:', error);
            this.showAlert('Failed to load pincode details');
        }
    },

    /**
     * Close modal
     */
    closeModal() {
        document.getElementById('pincode-modal').style.display = 'none';
    },

    /**
     * Show alert banner
     */
    showAlert(message) {
        document.getElementById('alert-message').textContent = message;
        document.getElementById('alert-banner').style.display = 'flex';
    },

    /**
     * Update anomaly badge
     */
    updateAnomalyBadge(count) {
        const badge = document.getElementById('anomaly-badge');
        if (badge) {
            badge.textContent = count;
            badge.style.display = count > 0 ? 'inline' : 'none';
        }
    },

    /**
     * Populate state filter dropdown
     */
    populateStateFilter() {
        const select = document.getElementById('state-filter');
        if (!select) return;

        select.innerHTML = '<option value="">All States</option>' +
            this.states.map(s => `<option value="${s}">${s}</option>`).join('');
    },

    /**
     * Filter by state
     */
    async filterByState(state) {
        try {
            const data = await API.getPincodes({ state, limit: 100 });
            Tables.renderFullPriorityTable(data.pincodes);
        } catch (error) {
            console.error('Failed to filter by state:', error);
        }
    },

    /**
     * Filter by risk
     */
    async filterByRisk(riskCategory) {
        try {
            const data = await API.getPincodes({ riskCategory, limit: 100 });
            Tables.renderFullPriorityTable(data.pincodes);
        } catch (error) {
            console.error('Failed to filter by risk:', error);
        }
    },

    /**
     * Toggle theme
     */
    toggleTheme() {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', next);

        const icon = document.querySelector('#theme-toggle i');
        icon.className = next === 'light' ? 'fas fa-sun' : 'fas fa-moon';

        localStorage.setItem('theme', next);
    },

    /**
     * Refresh all data
     */
    async refresh() {
        const btn = document.getElementById('refresh-btn');
        btn.querySelector('i').classList.add('fa-spin');

        await this.loadDashboardData();
        await this.loadPageData(this.currentPage);

        setTimeout(() => {
            btn.querySelector('i').classList.remove('fa-spin');
        }, 500);
    },

    /**
     * Check API health
     */
    async checkHealth() {
        try {
            const health = await API.getHealth();
            if (health.status !== 'healthy') {
                this.showAlert('API is running in degraded mode');
            }
        } catch (error) {
            console.warn('API health check failed');
        }
    },

    /**
     * Load demo data when API is unavailable
     */
    loadDemoData() {
        // Demo stats
        const demoStats = {
            critical_count: 47,
            high_count: 156,
            medium_count: 432,
            low_count: 1284,
            new_anomalies_today: 3,
            model_accuracy: 0.87,
            data_freshness_hours: 2
        };

        this.updateStats(demoStats);
        Charts.updateRiskChart(demoStats);

        // Demo priority table
        const demoPincodes = [
            { pincode: '843324', state: 'Bihar', overall_risk_score: 87, risk_category: 'CRITICAL', total_bio_updates: 636 },
            { pincode: '201102', state: 'Uttar Pradesh', overall_risk_score: 82, risk_category: 'CRITICAL', total_bio_updates: 523 },
            { pincode: '400001', state: 'Maharashtra', overall_risk_score: 76, risk_category: 'HIGH', total_bio_updates: 412 },
            { pincode: '302001', state: 'Rajasthan', overall_risk_score: 71, risk_category: 'HIGH', total_bio_updates: 389 },
            { pincode: '560001', state: 'Karnataka', overall_risk_score: 65, risk_category: 'HIGH', total_bio_updates: 345 }
        ];

        Tables.renderPriorityTable(demoPincodes);
    }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    // Load saved theme
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);

    // Initialize app
    App.init();
});

// Export
window.App = App;
