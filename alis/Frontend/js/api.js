/**
 * ALIS API Client
 * Handles all communication with the backend API
 */

const API = {
    // Base URL - configure based on environment
    baseUrl: 'http://localhost:8000/api/v1',

    /**
     * Generic fetch wrapper with error handling
     */
    async fetch(endpoint, options = {}) {
        try {
            const url = `${this.baseUrl}${endpoint}`;
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });

            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                throw new Error(error.detail || `HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`API Error [${endpoint}]:`, error);
            throw error;
        }
    },

    // ========== Analytics ==========

    async getDashboardStats() {
        return this.fetch('/analytics/dashboard-stats');
    },

    async getStateOverview() {
        return this.fetch('/analytics/state-overview');
    },

    async getClusterAnalysis() {
        return this.fetch('/analytics/cluster-analysis');
    },

    async getTrendAnalysis(days = 30, state = null) {
        let url = `/analytics/trend-analysis?days=${days}`;
        if (state) url += `&state=${encodeURIComponent(state)}`;
        return this.fetch(url);
    },

    // ========== Pincodes ==========

    async getPincodes(options = {}) {
        const params = new URLSearchParams();
        if (options.state) params.append('state', options.state);
        if (options.riskCategory) params.append('risk_category', options.riskCategory);
        if (options.skip) params.append('skip', options.skip);
        if (options.limit) params.append('limit', options.limit);

        const query = params.toString() ? `?${params.toString()}` : '';
        return this.fetch(`/pincodes${query}`);
    },

    async getMapData(state = null) {
        const query = state ? `?state=${encodeURIComponent(state)}` : '';
        return this.fetch(`/pincodes/map-data${query}`);
    },

    async getPriorityPincodes(limit = 20, urgency = null) {
        let url = `/pincodes/priority?limit=${limit}`;
        if (urgency) url += `&urgency=${urgency}`;
        return this.fetch(url);
    },

    async getStates() {
        return this.fetch('/pincodes/states');
    },

    async getPincodeDetail(pincode) {
        return this.fetch(`/pincodes/${pincode}`);
    },

    async getPincodeHistory(pincode, days = 90) {
        return this.fetch(`/pincodes/${pincode}/history?days=${days}`);
    },

    async getPincodeForecast(pincode, metricType = 'bio', horizon = 30) {
        return this.fetch(`/pincodes/${pincode}/forecast?metric_type=${metricType}&horizon=${horizon}`);
    },

    async getSimilarPincodes(pincode, limit = 10) {
        return this.fetch(`/pincodes/${pincode}/similar?limit=${limit}`);
    },

    async getPincodeRecommendations(pincode, status = null) {
        const query = status ? `?status=${status}` : '';
        return this.fetch(`/pincodes/${pincode}/recommendations${query}`);
    },

    // ========== Anomalies ==========

    async getAnomalies(options = {}) {
        const params = new URLSearchParams();
        if (options.days) params.append('days', options.days);
        if (options.severity) params.append('severity', options.severity);
        if (options.metricType) params.append('metric_type', options.metricType);
        if (options.skip) params.append('skip', options.skip);
        if (options.limit) params.append('limit', options.limit);

        const query = params.toString() ? `?${params.toString()}` : '';
        return this.fetch(`/anomalies${query}`);
    },

    async getRecentAnomalies(limit = 10) {
        return this.fetch(`/anomalies/recent?limit=${limit}`);
    },

    async getAnomalyDetail(anomalyId) {
        return this.fetch(`/anomalies/${anomalyId}`);
    },

    async investigateAnomaly(anomalyId, data) {
        return this.fetch(`/anomalies/${anomalyId}/investigate`, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    },

    async getAnomalyStats(days = 7) {
        return this.fetch(`/anomalies/stats/summary?days=${days}`);
    },

    // ========== Predictions ==========

    async getPredictions(pincode, days = 30, metricType = null) {
        let url = `/predictions/${pincode}?days=${days}`;
        if (metricType) url += `&metric_type=${metricType}`;
        return this.fetch(url);
    },

    async generatePredictions(pincode, days = 30, metricType = 'bio') {
        return this.fetch(`/predictions/${pincode}/generate?days=${days}&metric_type=${metricType}`, {
            method: 'POST'
        });
    },

    async validatePredictions(pincode) {
        return this.fetch(`/predictions/${pincode}/validate`);
    },

    async detectPeaks(pincode, days = 30, metricType = 'bio') {
        return this.fetch(`/predictions/peak-detection/${pincode}?days=${days}&metric_type=${metricType}`);
    },

    // ========== Health ==========

    async getHealth() {
        try {
            const response = await fetch(`${this.baseUrl.replace('/api/v1', '')}/health`);
            return await response.json();
        } catch (error) {
            return { status: 'offline', error: error.message };
        }
    }
};

// Export for use in other modules
window.API = API;
