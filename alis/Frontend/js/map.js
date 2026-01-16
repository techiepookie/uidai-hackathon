/**
 * ALIS Map Module
 * Leaflet-based India map with pincode visualization
 */

const MapView = {
    map: null,
    markers: null,

    // India center coordinates
    indiaCenter: [20.5937, 78.9629],
    defaultZoom: 5,

    // Risk color mapping
    riskColors: {
        CRITICAL: '#dc2626',
        HIGH: '#f97316',
        MEDIUM: '#eab308',
        LOW: '#22c55e'
    },

    /**
     * Initialize the map
     */
    init(containerId = 'india-map') {
        const container = document.getElementById(containerId);
        if (!container) return;

        // Check if map already initialized
        if (this.map) {
            this.map.remove();
        }

        // Create map
        this.map = L.map(containerId, {
            center: this.indiaCenter,
            zoom: this.defaultZoom,
            minZoom: 4,
            maxZoom: 15,
            zoomControl: true
        });

        // Add tile layer (dark theme)
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
            subdomains: 'abcd',
            maxZoom: 20
        }).addTo(this.map);

        // Initialize marker cluster group
        this.markers = L.markerClusterGroup({
            maxClusterRadius: 50,
            spiderfyOnMaxZoom: true,
            showCoverageOnHover: false,
            zoomToBoundsOnClick: true,
            disableClusteringAtZoom: 12,
            iconCreateFunction: this.createClusterIcon.bind(this)
        });

        this.map.addLayer(this.markers);

        // Fix map rendering issues
        setTimeout(() => {
            this.map.invalidateSize();
        }, 100);
    },

    /**
     * Create custom cluster icon
     */
    createClusterIcon(cluster) {
        const childCount = cluster.getChildCount();

        // Get average risk of cluster
        let totalRisk = 0;
        cluster.getAllChildMarkers().forEach(marker => {
            totalRisk += marker.options.riskScore || 0;
        });
        const avgRisk = totalRisk / childCount;

        // Determine color based on risk
        let color = this.riskColors.LOW;
        if (avgRisk >= 80) color = this.riskColors.CRITICAL;
        else if (avgRisk >= 60) color = this.riskColors.HIGH;
        else if (avgRisk >= 40) color = this.riskColors.MEDIUM;

        return L.divIcon({
            html: `<div style="
                background: ${color};
                width: 40px;
                height: 40px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: 600;
                font-size: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            ">${childCount}</div>`,
            className: 'cluster-icon',
            iconSize: [40, 40]
        });
    },

    /**
     * Create marker icon based on risk
     */
    createMarkerIcon(riskScore, category) {
        const color = this.riskColors[category] || this.riskColors.LOW;
        const size = riskScore >= 80 ? 14 : riskScore >= 60 ? 12 : 10;

        return L.divIcon({
            html: `<div style="
                background: ${color};
                width: ${size}px;
                height: ${size}px;
                border-radius: 50%;
                border: 2px solid rgba(255,255,255,0.5);
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            "></div>`,
            className: 'pincode-marker',
            iconSize: [size, size]
        });
    },

    /**
     * Load map data from API
     */
    async loadData(state = null) {
        try {
            const data = await API.getMapData(state);
            this.displayMarkers(data);
        } catch (error) {
            console.error('Failed to load map data:', error);
        }
    },

    /**
     * Display markers on map
     */
    displayMarkers(pincodes) {
        // Clear existing markers
        this.markers.clearLayers();

        // Since we don't have actual coordinates in the sample data,
        // we'll generate approximate coordinates based on pincode prefix
        const stateCoords = {
            'Andhra Pradesh': [15.9129, 79.7400],
            'Bihar': [25.0961, 85.3131],
            'Delhi': [28.7041, 77.1025],
            'Gujarat': [22.2587, 71.1924],
            'Karnataka': [15.3173, 75.7139],
            'Kerala': [10.8505, 76.2711],
            'Madhya Pradesh': [22.9734, 78.6569],
            'Maharashtra': [19.7515, 75.7139],
            'Rajasthan': [27.0238, 74.2179],
            'Tamil Nadu': [11.1271, 78.6569],
            'Uttar Pradesh': [26.8467, 80.9462],
            'West Bengal': [22.9868, 87.8550]
        };

        pincodes.forEach(p => {
            // Get base coordinates for state or use random India location
            let [lat, lng] = stateCoords[p.state] || this.indiaCenter;

            // Add some randomness to spread markers
            lat += (Math.random() - 0.5) * 2;
            lng += (Math.random() - 0.5) * 2;

            const marker = L.marker([lat, lng], {
                icon: this.createMarkerIcon(p.overall_risk_score, p.risk_category),
                riskScore: p.overall_risk_score
            });

            // Create popup content
            const popupContent = `
                <div style="min-width: 200px; font-family: Inter, sans-serif;">
                    <h4 style="margin: 0 0 8px; color: #1e293b;">${p.pincode}</h4>
                    <p style="margin: 0 0 4px; color: #64748b;">${p.state}</p>
                    <div style="display: flex; justify-content: space-between; margin-top: 12px;">
                        <div>
                            <div style="color: #64748b; font-size: 12px;">Risk Score</div>
                            <div style="font-weight: 600; color: ${this.riskColors[p.risk_category]};">
                                ${p.overall_risk_score.toFixed(1)}
                            </div>
                        </div>
                        <div>
                            <div style="color: #64748b; font-size: 12px;">Category</div>
                            <div style="font-weight: 600;">${p.risk_category}</div>
                        </div>
                    </div>
                    <button onclick="App.showPincodeDetail('${p.pincode}')" 
                            style="width: 100%; margin-top: 12px; padding: 8px; 
                                   background: #6366f1; color: white; border: none; 
                                   border-radius: 6px; cursor: pointer;">
                        View Details
                    </button>
                </div>
            `;

            marker.bindPopup(popupContent);
            this.markers.addLayer(marker);
        });

        // Fit bounds if we have markers
        if (pincodes.length > 0) {
            try {
                this.map.fitBounds(this.markers.getBounds(), { padding: [50, 50] });
            } catch (e) {
                // Reset to India view if bounds fail
                this.map.setView(this.indiaCenter, this.defaultZoom);
            }
        }
    },

    /**
     * Highlight a specific pincode
     */
    highlightPincode(pincode) {
        this.markers.eachLayer(marker => {
            if (marker.options.pincode === pincode) {
                marker.openPopup();
                this.map.setView(marker.getLatLng(), 12);
            }
        });
    },

    /**
     * Resize map (call after container size changes)
     */
    resize() {
        if (this.map) {
            this.map.invalidateSize();
        }
    }
};

// Export
window.MapView = MapView;
