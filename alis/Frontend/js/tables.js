/**
 * ALIS Tables Module
 * Data table rendering and management
 */

const Tables = {
    /**
     * Render priority table rows
     */
    renderPriorityTable(pincodes, tableBodyId = 'priority-tbody') {
        const tbody = document.getElementById(tableBodyId);
        if (!tbody) return;

        if (!pincodes || pincodes.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="7" class="loading-cell">
                        No data available
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = pincodes.map((p, index) => `
            <tr>
                <td><strong>${index + 1}</strong></td>
                <td>
                    <a href="#" onclick="App.showPincodeDetail('${p.pincode}'); return false;" 
                       style="color: var(--primary); text-decoration: none; font-weight: 500;">
                        ${p.pincode}
                    </a>
                </td>
                <td>${p.state}</td>
                <td>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <div style="width: 60px; height: 6px; background: var(--bg-tertiary); border-radius: 3px; overflow: hidden;">
                            <div style="width: ${p.overall_risk_score}%; height: 100%; background: ${this.getRiskColor(p.risk_category)};"></div>
                        </div>
                        <span>${p.overall_risk_score?.toFixed(1) || 0}</span>
                    </div>
                </td>
                <td>
                    <span class="risk-badge ${p.risk_category?.toLowerCase()}">${p.risk_category}</span>
                </td>
                <td>${this.formatNumber(p.total_bio_updates)}</td>
                <td>
                    <button class="btn-small" onclick="App.showPincodeDetail('${p.pincode}')">
                        <i class="fas fa-eye"></i> View
                    </button>
                </td>
            </tr>
        `).join('');
    },

    /**
     * Render full priority table with more columns
     */
    renderFullPriorityTable(pincodes, tableBodyId = 'full-priority-tbody') {
        const tbody = document.getElementById(tableBodyId);
        if (!tbody) return;

        if (!pincodes || pincodes.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="10" class="loading-cell">
                        No data available
                    </td>
                </tr>
            `;
            return;
        }

        tbody.innerHTML = pincodes.map((p, index) => `
            <tr>
                <td><strong>${index + 1}</strong></td>
                <td>
                    <a href="#" onclick="App.showPincodeDetail('${p.pincode}'); return false;" 
                       style="color: var(--primary); text-decoration: none; font-weight: 500;">
                        ${p.pincode}
                    </a>
                </td>
                <td>${p.state}</td>
                <td>${p.district || '-'}</td>
                <td>
                    <strong style="color: ${this.getRiskColor(p.risk_category)};">
                        ${p.overall_risk_score?.toFixed(1) || 0}
                    </strong>
                </td>
                <td>
                    <span class="risk-badge ${p.risk_category?.toLowerCase()}">${p.risk_category}</span>
                </td>
                <td>${p.bio_risk_score?.toFixed(1) || 0}</td>
                <td>${p.demo_risk_score?.toFixed(1) || 0}</td>
                <td>${p.migration_score?.toFixed(1) || 0}</td>
                <td>
                    <button class="btn-small" onclick="App.showPincodeDetail('${p.pincode}')">
                        <i class="fas fa-eye"></i> Details
                    </button>
                </td>
            </tr>
        `).join('');
    },

    /**
     * Render anomaly cards
     */
    renderAnomalyCards(anomalies, containerId = 'anomaly-cards') {
        const container = document.getElementById(containerId);
        if (!container) return;

        if (!anomalies || anomalies.length === 0) {
            container.innerHTML = `
                <div style="text-align: center; padding: 3rem; color: var(--text-muted);">
                    <i class="fas fa-check-circle" style="font-size: 3rem; margin-bottom: 1rem;"></i>
                    <p>No anomalies detected</p>
                </div>
            `;
            return;
        }

        container.innerHTML = anomalies.map(a => `
            <div class="anomaly-card">
                <div class="anomaly-header">
                    <h4>${a.pincode}</h4>
                    <span class="risk-badge ${a.severity?.toLowerCase()}">${a.severity}</span>
                </div>
                <div class="anomaly-metric">
                    <i class="fas ${this.getMetricIcon(a.metric_affected)}"></i>
                    ${this.formatMetricName(a.metric_affected)} - ${a.anomaly_type}
                </div>
                <div class="anomaly-stats">
                    <div class="anomaly-stat">
                        <div class="anomaly-stat-label">Expected</div>
                        <div class="anomaly-stat-value">${a.expected_value?.toFixed(0) || '-'}</div>
                    </div>
                    <div class="anomaly-stat">
                        <div class="anomaly-stat-label">Actual</div>
                        <div class="anomaly-stat-value">${a.actual_value?.toFixed(0)}</div>
                    </div>
                    <div class="anomaly-stat">
                        <div class="anomaly-stat-label">Deviation</div>
                        <div class="anomaly-stat-value" style="color: ${a.deviation_percent > 0 ? 'var(--danger)' : 'var(--success)'};">
                            ${a.deviation_percent > 0 ? '+' : ''}${a.deviation_percent?.toFixed(0)}%
                        </div>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: var(--text-muted); font-size: 0.875rem;">
                        Confidence: ${(a.confidence_score * 100).toFixed(0)}%
                    </span>
                    <span style="color: var(--text-muted); font-size: 0.875rem;">
                        ${new Date(a.detected_date).toLocaleDateString()}
                    </span>
                </div>
            </div>
        `).join('');
    },

    /**
     * Render pincode detail modal content
     */
    renderPincodeDetail(pincode, metric) {
        return `
            <div class="pincode-detail">
                <div class="detail-header" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
                    <div class="detail-stat">
                        <div style="color: var(--text-muted); font-size: 0.875rem;">Overall Risk</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: ${this.getRiskColor(metric.risk_category)};">
                            ${metric.overall_risk_score?.toFixed(1)}
                        </div>
                    </div>
                    <div class="detail-stat">
                        <div style="color: var(--text-muted); font-size: 0.875rem;">Bio Risk</div>
                        <div style="font-size: 1.5rem; font-weight: 700;">${metric.bio_risk_score?.toFixed(1)}</div>
                    </div>
                    <div class="detail-stat">
                        <div style="color: var(--text-muted); font-size: 0.875rem;">Demo Risk</div>
                        <div style="font-size: 1.5rem; font-weight: 700;">${metric.demo_risk_score?.toFixed(1)}</div>
                    </div>
                    <div class="detail-stat">
                        <div style="color: var(--text-muted); font-size: 0.875rem;">Migration</div>
                        <div style="font-size: 1.5rem; font-weight: 700;">${metric.migration_score?.toFixed(1)}</div>
                    </div>
                </div>
                
                <div class="detail-section" style="margin-bottom: 1.5rem;">
                    <h4 style="margin-bottom: 0.75rem;">Location</h4>
                    <p><strong>State:</strong> ${metric.state}</p>
                    <p><strong>District:</strong> ${metric.district || 'N/A'}</p>
                </div>
                
                <div class="detail-section" style="margin-bottom: 1.5rem;">
                    <h4 style="margin-bottom: 0.75rem;">Statistics</h4>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.5rem;">
                        <p><strong>Bio Updates:</strong> ${this.formatNumber(metric.total_bio_updates)}</p>
                        <p><strong>Demo Updates:</strong> ${this.formatNumber(metric.total_demo_updates)}</p>
                        <p><strong>Mobile Updates:</strong> ${this.formatNumber(metric.total_mobile_updates)}</p>
                        <p><strong>Enrollments:</strong> ${this.formatNumber(metric.total_enrollments)}</p>
                    </div>
                </div>
                
                <div class="detail-section" style="margin-bottom: 1.5rem;">
                    <h4 style="margin-bottom: 0.75rem;">Trends</h4>
                    <p><strong>Bio Trend:</strong> ${metric.bio_trend} 
                        ${metric.bio_trend === 'INCREASING' ? '📈' : metric.bio_trend === 'DECREASING' ? '📉' : '➡️'}
                    </p>
                    <p><strong>Demo Trend:</strong> ${metric.demo_trend}
                        ${metric.demo_trend === 'INCREASING' ? '📈' : metric.demo_trend === 'DECREASING' ? '📉' : '➡️'}
                    </p>
                </div>
                
                <div class="detail-section">
                    <h4 style="margin-bottom: 0.75rem;">Data Quality</h4>
                    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                        <span>Confidence:</span>
                        <div style="flex: 1; height: 8px; background: var(--bg-tertiary); border-radius: 4px; overflow: hidden;">
                            <div style="width: ${(metric.data_confidence * 100)}%; height: 100%; background: var(--success);"></div>
                        </div>
                        <span>${(metric.data_confidence * 100).toFixed(0)}%</span>
                    </div>
                    <p><strong>Days of Data:</strong> ${metric.days_of_data}</p>
                    <p><strong>Last Update:</strong> ${metric.last_update_date || 'N/A'}</p>
                </div>
            </div>
        `;
    },

    // Utility functions
    getRiskColor(category) {
        const colors = {
            CRITICAL: '#dc2626',
            HIGH: '#f97316',
            MEDIUM: '#eab308',
            LOW: '#22c55e'
        };
        return colors[category] || colors.LOW;
    },

    formatNumber(num) {
        if (!num) return '0';
        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
        if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
        return num.toString();
    },

    getMetricIcon(metric) {
        const icons = {
            bio: 'fa-fingerprint',
            demo: 'fa-id-card',
            mobile: 'fa-mobile-alt'
        };
        return icons[metric] || 'fa-chart-bar';
    },

    formatMetricName(metric) {
        const names = {
            bio: 'Biometric',
            demo: 'Demographic',
            mobile: 'Mobile'
        };
        return names[metric] || metric;
    }
};

// Export
window.Tables = Tables;
