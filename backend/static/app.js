"use strict";

/* ============================================================
   AEGISBURN AI
   FRONTEND APPLICATION
   ============================================================ */

const API_BASE = window.location.origin;

const state = {
    components: [],
    selectedComponentId: null,
    selectedResult: null,
    chart: null
};


/* ============================================================
   DOM HELPERS
   ============================================================ */

function $(id) {
    return document.getElementById(id);
}

function setText(id, value) {
    const element = $(id);

    if (!element) {
        return;
    }

    element.textContent =
        value === undefined || value === null
            ? ""
            : String(value);
}

function setHTML(id, value) {
    const element = $(id);

    if (!element) {
        return;
    }

    element.innerHTML = value;
}

function numberValue(value) {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {
        return null;
    }

    const number = Number(value);

    return Number.isFinite(number)
        ? number
        : null;
}

function formatNumber(value, decimals = 2) {
    const number = numberValue(value);

    if (number === null) {
        return "—";
    }

    return number.toLocaleString(undefined, {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

function formatValue(value, unit = "", decimals = 2) {
    const number = numberValue(value);

    if (number === null) {
        return "—";
    }

    return `${formatNumber(number, decimals)}${unit ? ` ${unit}` : ""}`;
}

function escapeHTML(value) {
    if (value === undefined || value === null) {
        return "";
    }

    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}


/* ============================================================
   RISK NORMALIZATION
   ============================================================ */

function normalizeRisk(value) {
    if (value === undefined || value === null || value === "") {
        return "REVIEW";
    }

    const risk = String(value)
        .trim()
        .toUpperCase();

    if (
        risk.includes("CRITICAL") ||
        risk === "CRIT"
    ) {
        return "CRITICAL";
    }

    if (
        risk.includes("HIGH RISK") ||
        risk === "HIGH" ||
        risk.includes("HIGH")
    ) {
        return "HIGH RISK";
    }

    if (
        risk.includes("MEDIUM") ||
        risk.includes("REVIEW")
    ) {
        return "REVIEW";
    }

    if (
        risk.includes("NORMAL") ||
        risk.includes("LOW")
    ) {
        return "NORMAL";
    }

    return "REVIEW";
}


function riskClass(risk) {
    const normalized = normalizeRisk(risk);

    if (normalized === "CRITICAL") {
        return "critical";
    }

    if (normalized === "HIGH RISK") {
        return "high-risk";
    }

    if (normalized === "NORMAL") {
        return "normal";
    }

    return "review";
}


function riskColors(risk) {
    const normalized = normalizeRisk(risk);

    if (normalized === "CRITICAL") {
        return {
            color: "#ff4d57",
            background: "rgba(255, 77, 87, 0.10)",
            border: "#ff4d57"
        };
    }

    if (normalized === "HIGH RISK") {
        return {
            color: "#ff7a18",
            background: "rgba(255, 122, 24, 0.10)",
            border: "#ff7a18"
        };
    }

    if (normalized === "NORMAL") {
        return {
            color: "#20c66b",
            background: "rgba(32, 198, 107, 0.10)",
            border: "#20c66b"
        };
    }

    return {
        color: "#f5b942",
        background: "rgba(245, 185, 66, 0.10)",
        border: "#f5b942"
    };
}


/* ============================================================
   API
   ============================================================ */

async function apiRequest(path, options = {}) {

    const fetchOptions = {
        method: options.method || "GET",
        headers: {
            ...(options.body
                ? {
                    "Content-Type": "application/json"
                }
                : {}),
            ...(options.headers || {})
        }
    };

    if (options.body !== undefined) {
        fetchOptions.body = JSON.stringify(
            options.body
        );
    }

    const response = await fetch(
        `${API_BASE}${path}`,
        fetchOptions
    );

    const contentType =
        response.headers.get("content-type") || "";

    let data;

    if (
        contentType.includes(
            "application/json"
        )
    ) {
        data = await response.json();
    } else {

        const text =
            await response.text();

        try {
            data = JSON.parse(text);
        } catch {
            data = {
                detail: text
            };
        }
    }

    if (!response.ok) {

        const message =
            data?.detail ||
            data?.message ||
            `Request failed with status ${response.status}`;

        throw new Error(
            typeof message === "string"
                ? message
                : JSON.stringify(message)
        );
    }

    return data;
}


/* ============================================================
   RESPONSE NORMALIZATION
   ============================================================ */

function extractComponents(data) {

    if (Array.isArray(data)) {
        return data;
    }

    if (Array.isArray(data?.components)) {
        return data.components;
    }

    if (Array.isArray(data?.data)) {
        return data.data;
    }

    if (Array.isArray(data?.items)) {
        return data.items;
    }

    return [];
}


function extractAnalysis(data) {

    if (!data) {
        return {};
    }

    if (
        data.result &&
        typeof data.result === "object"
    ) {
        return data.result;
    }

    if (
        data.analysis &&
        typeof data.analysis === "object"
    ) {
        return data.analysis;
    }

    if (
        data.data &&
        typeof data.data === "object" &&
        !Array.isArray(data.data)
    ) {
        return data.data;
    }

    return data;
}


/* ============================================================
   CONNECTION STATUS
   ============================================================ */

async function checkConnection() {

    const statusElement =
        $("systemStatus") ||
        $("connectionStatus");

    const statusDot =
        $("systemStatusDot");

    try {

        const health =
            await apiRequest("/health");

        if (statusElement) {

            statusElement.textContent =
                health.status === "healthy"
                    ? "System Online"
                    : "System Ready";

            statusElement.classList.remove(
                "offline",
                "connecting"
            );

            statusElement.classList.add(
                "online"
            );
        }

        if (statusDot) {

            statusDot.classList.remove(
                "offline",
                "connecting"
            );

            statusDot.classList.add(
                "online"
            );
        }

        return true;

    } catch (error) {

        console.error(
            "Backend connection failed:",
            error
        );

        if (statusElement) {

            statusElement.textContent =
                "Backend Offline";

            statusElement.classList.remove(
                "online",
                "connecting"
            );

            statusElement.classList.add(
                "offline"
            );
        }

        if (statusDot) {

            statusDot.classList.remove(
                "online",
                "connecting"
            );

            statusDot.classList.add(
                "offline"
            );
        }

        return false;
    }
}


/* ============================================================
   COMPONENTS
   ============================================================ */

async function loadComponents() {

    const response =
        await apiRequest("/components");

    const components =
        extractComponents(response);

    state.components =
        components;

    populateComponentSelector(
        components
    );

    updateDatasetInfo(
        components
    );

    updateSummary(
        components
    );

    return components;
}


function populateComponentSelector(
    components
) {

    const select =
        $("componentSelect") ||
        $("component-selector");

    if (!select) {

        console.warn(
            "Component selector not found."
        );

        return;
    }

    select.innerHTML = "";

    if (!components.length) {

        const option =
            document.createElement(
                "option"
            );

        option.value = "";

        option.textContent =
            "No components available";

        select.appendChild(
            option
        );

        return;
    }

    components.forEach(
        (component) => {

            const option =
                document.createElement(
                    "option"
                );

            const componentId =
                component.component_id ||
                component.id ||
                component.ComponentID ||
                "";

            const componentType =
                component.component_type ||
                component.type ||
                "";

            const parameter =
                component.parameter_name ||
                component.parameter ||
                "";

            option.value =
                componentId;

            option.textContent =
                `${componentId} — ${componentType} — ${parameter}`;

            select.appendChild(
                option
            );
        }
    );

    if (
        state.selectedComponentId &&
        components.some(
            component =>
                String(
                    component.component_id ||
                    component.id ||
                    component.ComponentID
                ) ===
                String(
                    state.selectedComponentId
                )
        )
    ) {

        select.value =
            state.selectedComponentId;

    } else {

        state.selectedComponentId =
            select.value;
    }
}


function findComponent(
    componentId
) {

    return state.components.find(
        (component) => {

            const id =
                component.component_id ||
                component.id ||
                component.ComponentID;

            return String(id) ===
                String(componentId);
        }
    );
}


/* ============================================================
   DATASET INFORMATION
   ============================================================ */

function updateDatasetInfo(
    components
) {

    setText(
        "datasetCount",
        components.length
            ? components.length.toLocaleString()
            : "0"
    );

    setText(
        "datasetStatus",
        components.length
            ? "Ready"
            : "No data"
    );

    setText(
        "datasetName",
        "Project Dataset"
    );
}


/* ============================================================
   SUMMARY
   ============================================================ */

function updateSummary(
    components
) {

    let normal = 0;
    let review = 0;
    let highRisk = 0;

    components.forEach(
        component => {

            const risk =
                normalizeRisk(
                    component.risk_level ||
                    component.risk ||
                    component.status
                );

            if (risk === "NORMAL") {
                normal++;
            } else if (
                risk === "HIGH RISK" ||
                risk === "CRITICAL"
            ) {
                highRisk++;
            } else {
                review++;
            }
        }
    );

    setText(
        "normalCount",
        normal.toLocaleString()
    );

    setText(
        "reviewCount",
        review.toLocaleString()
    );

    setText(
        "highRiskCount",
        highRisk.toLocaleString()
    );

    setText(
        "totalCount",
        components.length.toLocaleString()
    );
}


/* ============================================================
   ANALYZE COMPONENT
   ============================================================ */

async function analyzeComponent(
    componentId
) {

    if (!componentId) {
        return null;
    }

    state.selectedComponentId =
        componentId;

    setText(
        "componentStatus",
        `${componentId} AI screening in progress...`
    );

    /*
       IMPORTANT:
       Use GET /analyze/{component_id}
       first because this endpoint returns the
       complete analysis including:

       - anomaly
       - risk
       - predicted_168h
       - predicted_slope
       - engineering_limit
       - limit_utilization
       - reasons
    */

    try {

        const response =
            await apiRequest(
                `/analyze/${encodeURIComponent(
                    componentId
                )}`
            );

        const analysis =
            extractAnalysis(response);

        const baseComponent =
            findComponent(
                componentId
            ) || {};

        state.selectedResult = {
            ...baseComponent,
            ...analysis
        };

        renderAnalysis(
            state.selectedResult
        );

        setText(
            "componentStatus",
            `${componentId} AI screening complete.`
        );

        return state.selectedResult;

    } catch (getError) {

        console.warn(
            "GET /analyze/{component_id} failed:",
            getError
        );
    }


    /*
       Fallback to POST /analyze.
    */

    try {

        const response =
            await apiRequest(
                "/analyze",
                {
                    method: "POST",
                    body: {
                        component_id:
                            componentId
                    }
                }
            );

        const analysis =
            extractAnalysis(response);

        const baseComponent =
            findComponent(
                componentId
            ) || {};

        state.selectedResult = {
            ...baseComponent,
            ...analysis
        };

        renderAnalysis(
            state.selectedResult
        );

        setText(
            "componentStatus",
            `${componentId} AI screening complete.`
        );

        return state.selectedResult;

    } catch (postError) {

        console.warn(
            "POST /analyze failed:",
            postError
        );
    }


    /*
       Final fallback:
       local component data.

       This keeps the UI usable, but does NOT
       pretend that local raw data is an AI
       analysis.
    */

    const localComponent =
        findComponent(
            componentId
        );

    if (localComponent) {

        state.selectedResult = {
            ...localComponent
        };

        renderAnalysis(
            state.selectedResult
        );

        setText(
            "componentStatus",
            `${componentId} loaded from dataset.`
        );

        return state.selectedResult;
    }

    throw new Error(
        `Component ${componentId} could not be analyzed.`
    );
}


/* ============================================================
   COMPONENT INFORMATION
   ============================================================ */

function renderComponentInfo(
    data
) {

    const componentId =
        data.component_id ||
        data.id ||
        "—";

    const componentType =
        data.component_type ||
        data.type ||
        "—";

    const parameter =
        data.parameter_name ||
        data.parameter ||
        "—";

    const unit =
        cleanUnit(
            data.unit || ""
        );

    const lot =
        data.lot_id ||
        data.lot ||
        "—";

    const temperature =
        numberValue(
            data.temperature_c
        );

    setText(
        "componentId",
        componentId
    );

    setText(
        "componentType",
        componentType
    );

    setText(
        "parameterName",
        parameter
    );

    setText(
        "parameterUnit",
        unit || "—"
    );

    setText(
        "lotId",
        lot
    );

    setText(
        "temperature",
        temperature === null
            ? "—"
            : `${formatNumber(
                temperature,
                2
            )} °C`
    );
}


/* ============================================================
   CLEAN UNIT
   ============================================================ */

function cleanUnit(unit) {

    if (!unit) {
        return "";
    }

    const value =
        String(unit);

    /*
       Fix common UTF-8 display corruption
       such as ÂµA.
    */

    return value
        .replace(/ÂµA/g, "µA")
        .replace(/Âµ/g, "µ")
        .replace(/µA/g, "µA");
}


/* ============================================================
   COMPLETE RENDER
   ============================================================ */

function renderAnalysis(
    data
) {

    if (!data) {
        return;
    }

    renderComponentInfo(
        data
    );

    renderRiskPanel(
        data
    );

    renderAnomalyPanel(
        data
    );

    renderBehaviorPattern(
        data
    );

    renderPrediction(
        data
    );

    renderEngineeringLimit(
        data
    );

    renderReasons(
        data
    );

    renderGraph(
        data
    );

    renderOverview(
        data
    );
}


/* ============================================================
   RISK PANEL
   ============================================================ */

function renderRiskPanel(
    data
) {

    const riskScore =
        numberValue(
            data.risk_score ??
            data.riskScore
        );

    const riskLevel =
        normalizeRisk(
            data.risk_level ??
            data.risk ??
            data.status
        );

    const colors =
        riskColors(
            riskLevel
        );

    /*
       Risk score
    */

    setText(
        "riskScore",
        riskScore === null
            ? "—"
            : formatNumber(
                riskScore,
                1
            )
    );


    /*
       Risk badge
    */

    const badge =
        $("riskBadge");

    if (badge) {

        badge.textContent =
            riskLevel;

        badge.classList.remove(
            "badge-normal",
            "badge-review",
            "badge-high",
            "badge-high-risk",
            "badge-critical",
            "normal",
            "review",
            "high-risk",
            "critical"
        );

        badge.classList.add(
            `badge-${riskClass(
                riskLevel
            )}`
        );

        /*
           Inline styling guarantees that the
           correct color appears even if the
           stylesheet has an older class name.
        */

        badge.style.color =
            colors.color;

        badge.style.borderColor =
            colors.border;

        badge.style.backgroundColor =
            colors.background;
    }


    /*
       Risk score number color
    */

    const scoreElement =
        $("riskScore");

    if (scoreElement) {

        scoreElement.style.color =
            colors.color;
    }


    /*
       Risk progress bar
    */

    const riskBar =
        $("riskBar");

    if (riskBar) {

        const safeScore =
            riskScore === null
                ? 0
                : Math.max(
                    0,
                    Math.min(
                        100,
                        riskScore
                    )
                );

        riskBar.style.width =
            `${safeScore}%`;

        riskBar.style.background =
            colors.color;

        riskBar.style.borderRadius =
            "999px";

        riskBar.style.transition =
            "width 0.35s ease, background 0.25s ease";
    }


    /*
       Decision text
    */

    let decision =
        data.risk_decision ||
        data.decision;

    if (!decision) {

        if (
            riskLevel ===
            "CRITICAL"
        ) {
            decision =
                "CRITICAL — REJECT / INVESTIGATE";

        } else if (
            riskLevel ===
            "HIGH RISK"
        ) {
            decision =
                "HIGH RISK — REVIEW / INVESTIGATE";

        } else if (
            riskLevel ===
            "REVIEW"
        ) {
            decision =
                "REVIEW — ADDITIONAL SCREENING RECOMMENDED";

        } else {
            decision =
                "NORMAL — ACCEPT";
        }
    }

    setText(
        "riskDecision",
        decision
    );
}


/* ============================================================
   ANOMALY PANEL
   ============================================================ */

function renderAnomalyPanel(
    data
) {

    const anomalyIndex =
        numberValue(
            data.anomaly_index ??
            data.anomaly_score ??
            data.anomalyIndex
        );

    const anomalyFlag =
        Number(data.anomaly_flag) === 1 ||
        data.anomaly_flag === true;

    const backendLabel =
        data.anomaly_label;

    let anomalyLabel =
        backendLabel ||
        (
            anomalyFlag
                ? "ANOMALY"
                : "NORMAL"
        );

    anomalyLabel =
        String(
            anomalyLabel
        ).toUpperCase();


    /*
       Anomaly badge
    */

    const badge =
        $("anomalyBadge");

    if (badge) {

        badge.textContent =
            anomalyLabel;

        badge.classList.remove(
            "badge-normal",
            "badge-anomaly",
            "badge-review",
            "normal",
            "anomaly",
            "review"
        );

        if (
            anomalyLabel.includes(
                "ANOMAL"
            )
        ) {

            badge.classList.add(
                "badge-anomaly"
            );

            badge.style.color =
                "#ff4d57";

            badge.style.borderColor =
                "#ff4d57";

            badge.style.backgroundColor =
                "rgba(255, 77, 87, 0.10)";

        } else {

            badge.classList.add(
                "badge-normal"
            );

            badge.style.color =
                "#20c66b";

            badge.style.borderColor =
                "#20c66b";

            badge.style.backgroundColor =
                "rgba(32, 198, 107, 0.10)";
        }
    }


    /*
       Anomaly index
    */

    setText(
        "anomalyScore",
        anomalyIndex === null
            ? "—"
            : formatNumber(
                anomalyIndex,
                2
            )
    );


    /*
       Early drift
    */

    const earlyDrift =
        numberValue(
            data.early_drift_index ??
            data.earlyDriftIndex
        );

    setText(
        "earlyDrift",
        earlyDrift === null
            ? (
                numberValue(
                    data.drift_0_24
                ) === null
                    ? "—"
                    : formatNumber(
                        data.drift_0_24,
                        2
                    )
            )
            : formatNumber(
                earlyDrift,
                2
            )
    );


    /*
       0h
    */

    const value0 =
        numberValue(
            data.value_0h
        );

    const unit =
        cleanUnit(
            data.unit || ""
        );

    setText(
        "value0h",
        value0 === null
            ? "—"
            : formatValue(
                value0,
                unit,
                2
            )
    );


    /*
       24h
    */

    const value24 =
        numberValue(
            data.value_24h
        );

    setText(
        "value24h",
        value24 === null
            ? "—"
            : formatValue(
                value24,
                unit,
                2
            )
    );
}


/* ============================================================
   PREDICTION
   ============================================================ */

function renderPrediction(
    data
) {

    const unit =
        cleanUnit(
            data.prediction_unit ||
            data.parameter_unit ||
            data.unit ||
            ""
        );


    /*
       168h predicted value
    */

    const predicted168 =
        numberValue(
            data.predicted_168h
        );

    setText(
        "predicted168h",
        predicted168 === null
            ? "—"
            : formatNumber(
                predicted168,
                2
            )
    );


    /*
       Prediction unit
    */

    setText(
        "predictionUnit",
        unit || "—"
    );


    /*
       Actual 168h if the HTML contains it
    */

    const actual168 =
        numberValue(
            data.value_168h
        );

    setText(
        "actual168h",
        actual168 === null
            ? "—"
            : formatValue(
                actual168,
                unit,
                2
            )
    );


    /*
       Future drift
    */

    let futureDrift =
        numberValue(
            data.future_drift_index ??
            data.futureDriftIndex
        );

    /*
       If backend does not provide a future
       drift index, calculate a simple
       normalized future drift from the
       prediction and 24h value.
    */

    if (
        futureDrift === null &&
        predicted168 !== null
    ) {

        const value24 =
            numberValue(
                data.value_24h
            );

        if (
            value24 !== null &&
            value24 !== 0
        ) {

            futureDrift =
                Math.abs(
                    predicted168 -
                    value24
                ) /
                Math.abs(
                    value24
                ) *
                100;
        }
    }

    setText(
        "futureDrift",
        futureDrift === null
            ? "—"
            : formatNumber(
                futureDrift,
                2
            )
    );


    /*
       Predicted slope
    */

    const predictedSlope =
        numberValue(
            data.predicted_slope
        );

    setText(
        "predictedSlope",
        predictedSlope === null
            ? "—"
            : formatNumber(
                predictedSlope,
                6
            )
    );


    /*
       Safety slope if present
    */

    const safetySlope =
        numberValue(
            data.safety_slope
        );

    setText(
        "safetySlope",
        safetySlope === null
            ? "—"
            : formatNumber(
                safetySlope,
                6
            )
    );
}


/* ============================================================
   ENGINEERING LIMIT
   ============================================================ */

function getEngineeringLimit(
    data
) {

    const direct =
        numberValue(
            data.engineering_limit
        );

    if (direct !== null) {
        return direct;
    }

    /*
       Prototype engineering limits.
       These match the current AegisBurn
       project dataset configuration.
    */

    const parameter =
        String(
            data.parameter_name ||
            ""
        ).toLowerCase();

    if (
        parameter.includes(
            "iddq"
        )
    ) {
        return 50;
    }

    if (
        parameter.includes(
            "standby"
        )
    ) {
        return 80;
    }

    if (
        parameter.includes(
            "leakage"
        )
    ) {
        return 25;
    }

    if (
        parameter.includes(
            "propagation"
        )
    ) {
        return 40;
    }

    return null;
}


function calculateLimitUtilization(
    data
) {

    const limit =
        getEngineeringLimit(
            data
        );

    if (
        limit === null ||
        limit <= 0
    ) {
        return null;
    }

    const values = [
        data.value_0h,
        data.value_24h,
        data.value_96h,
        data.value_168h,
        data.predicted_168h
    ]
        .map(Number)
        .filter(
            Number.isFinite
        );

    if (!values.length) {
        return null;
    }

    /*
       Use the highest measured/predicted
       value so the safety panel reflects
       the worst available condition.
    */

    const maximum =
        Math.max(
            ...values
        );

    return Math.max(
        0,
        Math.min(
            100,
            maximum /
            limit *
            100
        )
    );
}


function renderEngineeringLimit(
    data
) {

    const limit =
        getEngineeringLimit(
            data
        );

    const unit =
        cleanUnit(
            data.unit ||
            data.parameter_unit ||
            ""
        );


    /*
       Backend utilization is used when it
       contains a meaningful value.

       If backend returns 0 while the actual
       measured/predicted values are non-zero,
       calculate the display utilization.
    */

    let utilization =
        numberValue(
            data.limit_utilization
        );

    const calculated =
        calculateLimitUtilization(
            data
        );

    if (
        calculated !== null &&
        (
            utilization === null ||
            utilization === 0
        )
    ) {
        utilization =
            calculated;
    }


    /*
       Engineering limit
    */

    setText(
        "engineeringLimit",
        limit === null
            ? "—"
            : `${formatNumber(
                limit,
                2
            )}${unit ? ` ${unit}` : ""}`
    );


    /*
       Utilization percentage
    */

    setText(
        "limitUtilization",
        utilization === null
            ? "—"
            : `${formatNumber(
                utilization,
                2
            )}%`
    );


    /*
       Limit bar
    */

    const limitBar =
        $("limitBar");

    if (limitBar) {

        const percentage =
            utilization === null
                ? 0
                : Math.max(
                    0,
                    Math.min(
                        100,
                        utilization
                    )
                );

        limitBar.style.width =
            `${percentage}%`;

        let color =
            "#20c66b";

        if (
            percentage >= 90
        ) {
            color =
                "#ff4d57";
        } else if (
            percentage >= 70
        ) {
            color =
                "#ff7a18";
        } else if (
            percentage >= 50
        ) {
            color =
                "#f5b942";
        }

        limitBar.style.background =
            color;

        limitBar.style.borderRadius =
            "999px";

        limitBar.style.transition =
            "width 0.35s ease";
    }


    /*
       Limit bar label
    */

    if (utilization === null) {

        setText(
            "limitBarLabel",
            "Select a component to view engineering-limit utilization."
        );

    } else {

        setText(
            "limitBarLabel",
            `${formatNumber(
                utilization,
                2
            )}% of engineering limit`
        );
    }
}


/* ============================================================
   BEHAVIOR PATTERN CLASSIFICATION
   ============================================================ */

function renderBehaviorPattern(
    data
) {

    const pattern =
        String(
            data.behavior_pattern ||
            "NORMAL"
        ).toUpperCase();

    const label =
        data.behavior_pattern_label ||
        "Normal";

    const explanation =
        data.behavior_pattern_explanation ||
        "";

    const badge =
        $("behaviorPatternBadge");

    if (badge) {

        badge.textContent =
            pattern.replace(
                /_/g,
                " "
            );

        badge.classList.remove(
            "badge-normal",
            "badge-anomaly",
            "badge-review",
            "badge-latent",
            "badge-sudden",
            "badge-gradual",
            "badge-watch"
        );

        if (pattern === "LATENT_DEFECT") {

            badge.classList.add(
                "badge-latent"
            );

        } else if (pattern === "SUDDEN_ANOMALY") {

            badge.classList.add(
                "badge-sudden"
            );

        } else if (pattern === "GRADUAL_DRIFT") {

            badge.classList.add(
                "badge-gradual"
            );

        } else if (pattern === "WATCH") {

            badge.classList.add(
                "badge-watch"
            );

        } else {

            badge.classList.add(
                "badge-normal"
            );
        }
    }

    setText(
        "behaviorPatternLabel",
        label
    );

    setText(
        "behaviorPatternExplanation",
        explanation ||
        "No additional detail available."
    );

    const labelElement =
        $("behaviorPatternLabel");

    if (labelElement) {

        labelElement.classList.remove(
            "pattern-normal",
            "pattern-latent",
            "pattern-sudden",
            "pattern-gradual",
            "pattern-watch"
        );

        const patternClassMap = {
            LATENT_DEFECT: "pattern-latent",
            SUDDEN_ANOMALY: "pattern-sudden",
            GRADUAL_DRIFT: "pattern-gradual",
            WATCH: "pattern-watch",
            NORMAL: "pattern-normal"
        };

        labelElement.classList.add(
            patternClassMap[pattern] ||
            "pattern-normal"
        );
    }
}


/* ============================================================
   REASONS
   ============================================================ */

function renderReasons(
    data
) {

    const container =
        $("riskReasons") ||
        $("reasonsList");

    if (!container) {
        return;
    }

    let reasons = [];

    if (
        Array.isArray(
            data.risk_reasons
        )
    ) {
        reasons =
            data.risk_reasons;
    }

    if (
        Array.isArray(
            data.reasons
        ) &&
        data.reasons.length
    ) {
        reasons =
            data.reasons;
    }


    /*
       If backend does not provide reasons,
       create transparent reasons from the
       returned metrics.

       This is UI explanation only.
       It does not modify the backend risk.
    */

    if (!reasons.length) {

        const generated =
            generateReasons(
                data
            );

        reasons =
            generated;
    }


    if (!reasons.length) {

        container.innerHTML =
            `
            <div class="empty-reasons">
                No additional risk factors reported.
            </div>
            `;

        return;
    }


    container.innerHTML =
        reasons
            .map(
                reason =>
                    `
                    <div class="reason-item">
                        ${escapeHTML(
                            reason
                        )}
                    </div>
                    `
            )
            .join("");
}


function generateReasons(
    data
) {

    const reasons = [];

    const riskScore =
        numberValue(
            data.risk_score
        );

    const anomalyIndex =
        numberValue(
            data.anomaly_index ??
            data.anomaly_score
        );

    const value24 =
        numberValue(
            data.value_24h
        );

    const value168 =
        numberValue(
            data.value_168h
        );

    const predicted168 =
        numberValue(
            data.predicted_168h
        );

    const limit =
        getEngineeringLimit(
            data
        );


    if (
        riskScore !== null &&
        riskScore >= 80
    ) {

        reasons.push(
            `Overall risk score is ${formatNumber(
                riskScore,
                1
            )}/100.`
        );
    }


    if (
        anomalyIndex !== null &&
        anomalyIndex >= 70
    ) {

        reasons.push(
            `Early anomaly index is elevated at ${formatNumber(
                anomalyIndex,
                2
            )}.`
        );
    }


    if (
        value24 !== null &&
        value168 !== null &&
        value168 > value24
    ) {

        reasons.push(
            `Measured parameter increased from ${formatNumber(
                value24,
                2
            )} to ${formatNumber(
                value168,
                2
            )}.`
        );
    }


    if (
        predicted168 !== null &&
        limit !== null
    ) {

        const predictedUtilization =
            predicted168 /
            limit *
            100;

        if (
            predictedUtilization >= 70
        ) {

            reasons.push(
                `Predicted 168h value reaches ${formatNumber(
                    predictedUtilization,
                    1
                )}% of the engineering limit.`
            );
        }
    }


    if (
        data.defect_type &&
        String(
            data.defect_type
        ).toUpperCase() !==
        "NORMAL"
    ) {

        /*
           Dataset ground-truth labels are
           deliberately NOT used here.
           They are only shown if already
           supplied by the backend dataset.
        */

    }


    return reasons;
}


/* ============================================================
   OVERVIEW
   ============================================================ */

function renderOverview(
    data
) {

    const overviewBody =
        $("overviewBody") ||
        $("componentOverviewBody");

    if (!overviewBody) {
        return;
    }

    const componentId =
        data.component_id ||
        data.id ||
        "—";

    const componentType =
        data.component_type ||
        data.type ||
        "—";

    const parameter =
        data.parameter_name ||
        data.parameter ||
        "—";

    const lot =
        data.lot_id ||
        data.lot ||
        "—";

    const unit =
        cleanUnit(
            data.unit ||
            ""
        );

    const value0 =
        data.value_0h;

    const value24 =
        data.value_24h;

    const risk =
        normalizeRisk(
            data.risk_level ||
            data.risk
        );

    overviewBody.innerHTML =
        `
        <tr>
            <td>
                ${escapeHTML(
                    componentId
                )}
            </td>

            <td>
                ${escapeHTML(
                    componentType
                )}
            </td>

            <td>
                ${escapeHTML(
                    parameter
                )}
            </td>

            <td>
                ${escapeHTML(
                    lot
                )}
            </td>

            <td>
                ${escapeHTML(
                    formatValue(
                        value0,
                        unit,
                        2
                    )
                )}
            </td>

            <td>
                ${escapeHTML(
                    formatValue(
                        value24,
                        unit,
                        2
                    )
                )}
            </td>

            <td>
                <span
                    class="risk-badge ${riskClass(
                        risk
                    )}"
                >
                    ${escapeHTML(
                        risk
                    )}
                </span>
            </td>
        </tr>
        `;
}


/* ============================================================
   GRAPH
   ============================================================ */

function renderGraph(
    data
) {

    const container =
        $("burnInChart") ||
        $("trendChart") ||
        $("measurementChart");

    if (!container) {
        return;
    }

    const measurements = [
        {
            hour: 0,
            value:
                numberValue(
                    data.value_0h
                )
        },
        {
            hour: 24,
            value:
                numberValue(
                    data.value_24h
                )
        },
        {
            hour: 96,
            value:
                numberValue(
                    data.value_96h
                )
        },
        {
            hour: 168,
            value:
                numberValue(
                    data.value_168h
                )
        }
    ].filter(
        point =>
            point.value !== null
    );


    const predictedValue =
        numberValue(
            data.predicted_168h
        );


    if (!measurements.length) {

        if (
            state.chart &&
            typeof state.chart.destroy ===
            "function"
        ) {

            try {
                state.chart.destroy();
            } catch {
                // Ignore old chart cleanup errors.
            }

            state.chart = null;
        }

        container.innerHTML =
            `
            <div class="chart-empty">
                No measurement data available.
            </div>
            `;

        return;
    }


    /*
       container is a <div>. Use Chart.js on an
       internal canvas when available, otherwise
       fall back to an inline SVG rendered directly
       into the div (never into a canvas — canvas
       does not render its innerHTML).
    */

    if (
        typeof Chart !==
        "undefined"
    ) {

        let canvas =
            container.querySelector(
                "canvas"
            );

        if (!canvas) {

            container.innerHTML = "";

            canvas =
                document.createElement(
                    "canvas"
                );

            container.appendChild(
                canvas
            );
        }

        renderChartJS(
            canvas,
            measurements,
            predictedValue,
            data
        );

        return;
    }


    /*
       Chart.js not available (CDN blocked/offline).
       Render SVG straight into the div.
    */

    if (
        state.chart &&
        typeof state.chart.destroy ===
        "function"
    ) {

        try {
            state.chart.destroy();
        } catch {
            // Ignore old chart cleanup errors.
        }

        state.chart = null;
    }

    renderSVGChart(
        container,
        measurements,
        predictedValue,
        data
    );
}


/* ============================================================
   CHART.JS
   ============================================================ */

function renderChartJS(
    canvas,
    measurements,
    predictedValue,
    data
) {

    if (
        state.chart &&
        typeof state.chart.destroy ===
        "function"
    ) {

        try {
            state.chart.destroy();
        } catch {
            // Ignore old chart cleanup errors.
        }

        state.chart = null;
    }


    /*
       Always plot a fixed hour axis (0/24/96/168),
       not just the hours that have an actual
       measurement. This matters when value_168h is
       not yet known (pre-168h screening) — the AI
       prediction still needs a 168h slot to plot
       against, even though there is no actual point
       there yet.
    */

    const allHours = [
        0,
        24,
        96,
        168
    ];

    const labels =
        allHours.map(
            hour =>
                `${hour}h`
        );

    const valueByHour = {};

    measurements.forEach(
        point => {
            valueByHour[
                point.hour
            ] = point.value;
        }
    );

    const measuredData =
        allHours.map(
            hour =>
                valueByHour[hour] ??
                null
        );


    const predictedData =
        allHours.map(
            () => null
        );


    /*
       Bridge the dashed prediction line from the
       last known actual measurement through to the
       168h prediction, regardless of whether an
       actual 168h value exists.
    */

    if (
        predictedValue !== null
    ) {

        const index168 =
            allHours.indexOf(168);

        const hasValue =
            hour =>
                valueByHour[hour] !==
                undefined &&
                valueByHour[hour] !==
                null;

        /*
           Bridge from 24h when available (matches
           the SVG renderer's behavior), otherwise
           fall back to the earliest available early
           point. Never bridge from the 168h slot
           itself — even when an actual 168h value
           exists, the prediction should still show
           the forecast trajectory from earlier data,
           not connect a point to itself.
        */

        let bridgeIndex = -1;

        const bridgePreference = [
            96,
            24,
            0
        ];

        for (
            const hour of bridgePreference
        ) {

            if (hasValue(hour)) {

                bridgeIndex =
                    allHours.indexOf(
                        hour
                    );

                break;
            }
        }

        if (
            bridgeIndex >= 0 &&
            bridgeIndex !== index168
        ) {

            predictedData[
                bridgeIndex
            ] =
                measuredData[
                    bridgeIndex
                ];

            predictedData[
                index168
            ] = predictedValue;
        }
    }


    const context =
        canvas.getContext(
            "2d"
        );


    state.chart =
        new Chart(
            context,
            {
                type: "line",

                data: {
                    labels,

                    datasets: [
                        {
                            label:
                                "Measured Value",

                            data:
                                measuredData,

                            borderWidth:
                                2,

                            tension:
                                0.25,

                            pointRadius:
                                4,

                            fill:
                                false
                        },

                        {
                            label:
                                "AI Predicted Trend",

                            data:
                                predictedData,

                            borderWidth:
                                2,

                            borderDash:
                                [
                                    6,
                                    5
                                ],

                            spanGaps:
                                true,

                            tension:
                                0.25,

                            pointRadius:
                                4,

                            fill:
                                false
                        }
                    ]
                },

                options: {
                    responsive:
                        true,

                    maintainAspectRatio:
                        false,

                    interaction: {
                        mode:
                            "index",

                        intersect:
                            false
                    },

                    plugins: {
                        legend: {
                            display:
                                true
                        }
                    },

                    scales: {
                        x: {
                            title: {
                                display:
                                    true,

                                text:
                                    "Burn-In Time (Hours)"
                            }
                        },

                        y: {
                            title: {
                                display:
                                    true,

                                text:
                                    data.parameter_name ||
                                    "Measured Value"
                            }
                        }
                    }
                }
            }
        );
}


/* ============================================================
   SVG FALLBACK CHART
   ============================================================ */

function renderSVGChart(
    container,
    measurements,
    predictedValue,
    data
) {

    const width =
        1000;

    const height =
        420;

    const padding = {
        top: 45,
        right: 40,
        bottom: 65,
        left: 75
    };

    const chartWidth =
        width -
        padding.left -
        padding.right;

    const chartHeight =
        height -
        padding.top -
        padding.bottom;


    const allValues =
        measurements
            .map(
                point =>
                    point.value
            );

    if (
        predictedValue !== null
    ) {
        allValues.push(
            predictedValue
        );
    }


    let minValue =
        Math.min(
            ...allValues
        );

    let maxValue =
        Math.max(
            ...allValues
        );

    const range =
        maxValue -
        minValue;


    if (
        range === 0
    ) {

        minValue -= 1;
        maxValue += 1;

    } else {

        minValue -=
            range * 0.12;

        maxValue +=
            range * 0.12;
    }


    const xScale =
        hour =>
            padding.left +
            (
                hour /
                168
            ) *
            chartWidth;


    const yScale =
        value =>
            padding.top +
            (
                (
                    maxValue -
                    value
                ) /
                (
                    maxValue -
                    minValue
                )
            ) *
            chartHeight;


    const linePoints =
        measurements
            .map(
                point =>
                    `${xScale(
                        point.hour
                    )},${yScale(
                        point.value
                    )}`
            )
            .join(" ");


    const areaPoints = [
        `${xScale(
            measurements[0].hour
        )},${padding.top + chartHeight}`,

        ...measurements.map(
            point =>
                `${xScale(
                    point.hour
                )},${yScale(
                    point.value
                )}`
        ),

        `${xScale(
            measurements[
                measurements.length - 1
            ].hour
        )},${padding.top + chartHeight}`
    ].join(" ");


    let predictionLine =
        "";


    if (
        predictedValue !== null
    ) {

        const earlyMeasurements =
            measurements.filter(
                point =>
                    point.hour !== 168
            );

        const predictedStart =
            earlyMeasurements[
                earlyMeasurements.length - 1
            ] ||
            measurements[0];


        predictionLine =
            `
            <line
                x1="${xScale(
                    predictedStart.hour
                )}"
                y1="${yScale(
                    predictedStart.value
                )}"
                x2="${xScale(168)}"
                y2="${yScale(
                    predictedValue
                )}"
                class="prediction-line"
            />

            <circle
                cx="${xScale(168)}"
                cy="${yScale(
                    predictedValue
                )}"
                r="6"
                class="prediction-point"
            />

            <text
                x="${xScale(168)}"
                y="${yScale(
                    predictedValue
                ) - 14}"
                text-anchor="middle"
                class="chart-value-label"
            >
                AI ${formatNumber(
                    predictedValue,
                    2
                )}
            </text>
            `;
    }


    let gridLines =
        "";

    let yLabels =
        "";


    const gridCount =
        6;


    for (
        let i = 0;
        i <= gridCount;
        i++
    ) {

        const ratio =
            i /
            gridCount;

        const y =
            padding.top +
            ratio *
            chartHeight;

        const value =
            maxValue -
            ratio *
            (
                maxValue -
                minValue
            );


        gridLines +=
            `
            <line
                x1="${padding.left}"
                y1="${y}"
                x2="${width - padding.right}"
                y2="${y}"
                class="chart-grid"
            />
            `;


        yLabels +=
            `
            <text
                x="${padding.left - 15}"
                y="${y + 5}"
                text-anchor="end"
                class="chart-axis-label"
            >
                ${formatNumber(
                    value,
                    1
                )}
            </text>
            `;
    }


    let xLabels =
        "";


    const xTicks = [
        0,
        24,
        48,
        72,
        96,
        120,
        144,
        168
    ];


    xTicks.forEach(
        hour => {

            const x =
                xScale(
                    hour
                );


            xLabels +=
                `
                <line
                    x1="${x}"
                    y1="${padding.top}"
                    x2="${x}"
                    y2="${padding.top + chartHeight}"
                    class="chart-grid vertical"
                />

                <text
                    x="${x}"
                    y="${height - 30}"
                    text-anchor="middle"
                    class="chart-axis-label"
                >
                    ${hour}h
                </text>
                `;
        }
    );


    const circles =
        measurements
            .map(
                point =>
                    `
                    <circle
                        cx="${xScale(
                            point.hour
                        )}"
                        cy="${yScale(
                            point.value
                        )}"
                        r="6"
                        class="measurement-point"
                    >
                        <title>
                            ${point.hour}h:
                            ${formatValue(
                                point.value,
                                cleanUnit(
                                    data.unit ||
                                    ""
                                ),
                                2
                            )}
                        </title>
                    </circle>
                    `
            )
            .join("");


    container.innerHTML =
        `
        <svg
            class="burnin-svg"
            viewBox="0 0 ${width} ${height}"
            role="img"
            aria-label="Burn-in measurement and AI prediction chart"
        >

            ${gridLines}

            ${xLabels}

            ${yLabels}

            <polygon
                points="${areaPoints}"
                class="measurement-area"
            />

            <polyline
                points="${linePoints}"
                class="measurement-line"
            />

            ${predictionLine}

            ${circles}

            <text
                x="${width / 2}"
                y="${height - 5}"
                text-anchor="middle"
                class="chart-axis-title"
            >
                Burn-In Time (Hours)
            </text>

            <text
                x="22"
                y="${height / 2}"
                text-anchor="middle"
                transform="rotate(-90 22 ${height / 2})"
                class="chart-axis-title"
            >
                ${escapeHTML(
                    data.parameter_name ||
                    "Measured Value"
                )}
            </text>

        </svg>

        <div class="chart-legend">

            <div class="legend-item">
                <span
                    class="legend-line measured"
                ></span>
                Measured Value
            </div>

            <div class="legend-item">
                <span
                    class="legend-line predicted"
                ></span>
                AI Predicted Trend
            </div>

        </div>
        `;
}


/* ============================================================
   CSV UPLOAD
   ============================================================ */

function registerUploadEvents() {

    const dropzone =
        $("csvDropzone");

    const input =
        $("csvFileInput");

    const uploadButton =
        $("uploadCsvButton");

    if (!dropzone || !input) {
        return;
    }


    dropzone.addEventListener(
        "click",
        () => {
            input.click();
        }
    );


    dropzone.addEventListener(
        "keydown",
        event => {

            if (
                event.key ===
                "Enter" ||
                event.key ===
                " "
            ) {

                event.preventDefault();

                input.click();
            }
        }
    );


    input.addEventListener(
        "change",
        () => {

            const file =
                input.files?.[0];

            if (!file) {
                return;
            }

            setText(
                "csvFileLabel",
                file.name
            );

            if (uploadButton) {
                uploadButton.disabled =
                    false;
            }

            setText(
                "uploadStatus",
                `${file.name} selected.`
            );
        }
    );


    dropzone.addEventListener(
        "dragover",
        event => {

            event.preventDefault();

            dropzone.classList.add(
                "dragover"
            );
        }
    );


    dropzone.addEventListener(
        "dragleave",
        () => {

            dropzone.classList.remove(
                "dragover"
            );
        }
    );


    dropzone.addEventListener(
        "drop",
        event => {

            event.preventDefault();

            dropzone.classList.remove(
                "dragover"
            );

            const file =
                event.dataTransfer
                    ?.files?.[0];

            if (!file) {
                return;
            }

            if (
                !file.name
                    .toLowerCase()
                    .endsWith(
                        ".csv"
                    )
            ) {

                setText(
                    "uploadStatus",
                    "Please select a CSV file."
                );

                return;
            }

            input.files =
                event.dataTransfer.files;

            setText(
                "csvFileLabel",
                file.name
            );

            if (uploadButton) {
                uploadButton.disabled =
                    false;
            }

            setText(
                "uploadStatus",
                `${file.name} selected.`
            );
        }
    );


    if (uploadButton) {

        uploadButton.addEventListener(
            "click",
            uploadCSV
        );
    }
}


async function uploadCSV() {

    const input =
        $("csvFileInput");

    const button =
        $("uploadCsvButton");

    const status =
        $("uploadStatus");

    const file =
        input?.files?.[0];

    if (!file) {
        return;
    }


    try {

        if (button) {
            button.disabled =
                true;

            button.textContent =
                "Uploading...";
        }

        if (status) {
            status.textContent =
                "Uploading and processing CSV...";
        }


        const formData =
            new FormData();

        formData.append(
            "file",
            file
        );


        const response =
            await fetch(
                `${API_BASE}/upload-csv`,
                {
                    method: "POST",
                    body: formData
                }
            );


        const contentType =
            response.headers.get(
                "content-type"
            ) || "";


        let data;

        if (
            contentType.includes(
                "application/json"
            )
        ) {

            data =
                await response.json();

        } else {

            data = {
                detail:
                    await response.text()
            };
        }


        if (!response.ok) {

            let message =
                "CSV upload failed.";

            if (typeof data?.detail === "string") {

                message = data.detail;

            } else if (Array.isArray(data?.detail)) {

                message = data.detail
                    .map((item) =>
                        item?.msg ||
                        JSON.stringify(item)
                    )
                    .join("; ");

            } else if (data?.detail) {

                message =
                    JSON.stringify(data.detail);

            } else if (data?.message) {

                message = data.message;
            }

            throw new Error(message);
        }


        if (status) {

            status.textContent =
                "CSV uploaded successfully.";
        }


        await loadComponents();


        const select =
            $("componentSelect");

        if (
            select &&
            select.value
        ) {

            state.selectedComponentId =
                select.value;

            await analyzeComponent(
                select.value
            );
        }


    } catch (error) {

        console.error(
            "CSV upload failed:",
            error
        );

        if (status) {

            status.textContent =
                `Upload failed: ${error.message}`;
        }

    } finally {

        if (button) {

            button.disabled =
                false;

            button.textContent =
                "Upload & Analyze";
        }
    }
}


/* ============================================================
   REFRESH
   ============================================================ */

async function refreshDashboard() {

    const refreshButton =
        $("refreshButton") ||
        $("refresh");

    if (refreshButton) {

        refreshButton.disabled =
            true;

        refreshButton.textContent =
            "Refreshing...";
    }


    try {

        const online =
            await checkConnection();

        if (!online) {

            throw new Error(
                "Cannot connect to AegisBurn backend."
            );
        }


        const components =
            await loadComponents();


        const select =
            $("componentSelect") ||
            $("component-selector");


        if (
            components.length &&
            select
        ) {

            let componentId =
                state.selectedComponentId;


            if (
                !componentId ||
                !findComponent(
                    componentId
                )
            ) {

                componentId =
                    select.value;
            }


            if (componentId) {

                select.value =
                    componentId;

                await analyzeComponent(
                    componentId
                );
            }
        }


    } catch (error) {

        console.error(
            "Dashboard refresh failed:",
            error
        );

        showError(
            error.message
        );

    } finally {

        if (refreshButton) {

            refreshButton.disabled =
                false;

            refreshButton.textContent =
                "Refresh";
        }
    }
}


/* ============================================================
   ERROR
   ============================================================ */

function showError(
    message
) {

    const container =
        $("errorMessage") ||
        $("componentStatus");

    if (!container) {
        return;
    }

    container.textContent =
        `Error: ${message}`;
}


/* ============================================================
   EVENT LISTENERS
   ============================================================ */

function registerEvents() {

    const select =
        $("componentSelect") ||
        $("component-selector");


    if (select) {

        select.addEventListener(
            "change",
            async event => {

                const componentId =
                    event.target.value;

                if (!componentId) {
                    return;
                }

                await analyzeComponent(
                    componentId
                );
            }
        );
    }


    const analyzeButton =
        $("analyzeButton");


    if (analyzeButton) {

        analyzeButton.addEventListener(
            "click",
            async () => {

                const componentId =
                    state.selectedComponentId ||
                    select?.value;

                if (!componentId) {
                    return;
                }

                await analyzeComponent(
                    componentId
                );
            }
        );
    }


    const refreshButton =
        $("refreshButton") ||
        $("refresh");


    if (refreshButton) {

        refreshButton.addEventListener(
            "click",
            refreshDashboard
        );
    }


    registerUploadEvents();
}


/* ============================================================
   STARTUP
   ============================================================ */

async function initializeApp() {

    console.log(
        "AEGISBURN AI FRONTEND STARTING..."
    );


    const statusElement =
        $("systemStatus") ||
        $("connectionStatus");


    const statusDot =
        $("systemStatusDot");


    if (statusElement) {

        statusElement.textContent =
            "Connecting...";

        statusElement.classList.add(
            "connecting"
        );
    }


    if (statusDot) {

        statusDot.classList.add(
            "connecting"
        );
    }


    registerEvents();


    await refreshDashboard();


    console.log(
        "AEGISBURN AI FRONTEND READY"
    );
}


/* ============================================================
   DOM READY
   ============================================================ */

if (
    document.readyState ===
    "loading"
) {

    document.addEventListener(
        "DOMContentLoaded",
        initializeApp
    );

} else {

    initializeApp();
}