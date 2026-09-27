"use strict";

/* ==========================================================
   AEGISBURN AI — FRONTEND DASHBOARD
   ========================================================== */

const API_BASE = "http://127.0.0.1:8001";

let currentComponent = null;


/* ==========================================================
   DOM
   ========================================================== */

function $(selector) {
    return document.querySelector(selector);
}


/* ==========================================================
   API
   ========================================================== */

async function apiGet(path) {

    const controller =
        new AbortController();

    const timeout =
        setTimeout(
            () => controller.abort(),
            15000
        );

    try {

        const response =
            await fetch(
                `${API_BASE}${path}`,
                {
                    method: "GET",
                    signal: controller.signal,
                    cache: "no-store"
                }
            );

        if (!response.ok) {

            let message =
                `HTTP ${response.status}`;

            try {

                const data =
                    await response.json();

                if (data.detail) {
                    message = data.detail;
                }

            } catch (_) {}

            throw new Error(
                message
            );
        }

        return await response.json();

    } finally {

        clearTimeout(
            timeout
        );
    }
}


/* ==========================================================
   FORMATTERS
   ========================================================== */

function formatNumber(
    value,
    digits = 2
) {

    if (
        value === null ||
        value === undefined ||
        !Number.isFinite(
            Number(value)
        )
    ) {
        return "—";
    }

    return Number(value).toLocaleString(
        undefined,
        {
            maximumFractionDigits: digits
        }
    );
}


function riskClass(level) {

    const value =
        String(level || "")
            .trim()
            .toUpperCase();

    if (value === "HIGH RISK") {
        return "high";
    }

    if (value === "REVIEW") {
        return "review";
    }

    return "normal";
}


function escapeHtml(value) {

    return String(
        value ?? ""
    )
        .replaceAll(
            "&",
            "&amp;"
        )
        .replaceAll(
            "<",
            "&lt;"
        )
        .replaceAll(
            ">",
            "&gt;"
        )
        .replaceAll(
            '"',
            "&quot;"
        )
        .replaceAll(
            "'",
            "&#039;"
        );
}


/* ==========================================================
   HEALTH
   ========================================================== */

async function loadHealth() {

    const status =
        $("#systemStatus");

    try {

        const data =
            await apiGet(
                "/health"
            );

        if (status) {

            status.textContent =
                `Online • ${data.anomaly_models} anomaly + ${data.prediction_models} prediction models`;
        }

        return data;

    } catch (error) {

        if (status) {

            status.textContent =
                "Backend unavailable";
        }

        throw error;
    }
}


/* ==========================================================
   METADATA
   ========================================================== */

async function loadMetadata() {

    const data =
        await apiGet(
            "/metadata"
        );

    const total =
        $("#totalCount");

    if (total) {

        total.textContent =
            formatNumber(
                data.dataset.components,
                0
            );
    }

    /*
     * Current backend metadata does not return risk counts,
     * because we intentionally removed the expensive
     * 10,000-component startup analysis.
     *
     * Keep the overview cards clean instead of pretending
     * these numbers are already known.
     */

    const normal =
        $("#normalCount");

    const review =
        $("#reviewCount");

    const high =
        $("#highRiskCount");

    if (normal) {
        normal.textContent = "—";
    }

    if (review) {
        review.textContent = "—";
    }

    if (high) {
        high.textContent = "—";
    }

    return data;
}


/* ==========================================================
   COMPONENT ID SELECTOR
   ========================================================== */

function buildComponentSelector() {

    const select =
        $("#componentSelect");

    if (!select) {
        return;
    }

    select.innerHTML = "";

    const placeholder =
        document.createElement(
            "option"
        );

    placeholder.value = "";

    placeholder.textContent =
        "Select a component...";

    select.appendChild(
        placeholder
    );

    /*
     * The current dataset uses sequential component IDs:
     * CMP-000001, CMP-000002, ...
     *
     * We only populate the first 100 here so the browser
     * does not need to download 10,000 records.
     */

    for (
        let i = 1;
        i <= 100;
        i++
    ) {

        const option =
            document.createElement(
                "option"
            );

        const id =
            `CMP-${String(i).padStart(6, "0")}`;

        option.value = id;

        option.textContent = id;

        select.appendChild(
            option
        );
    }
}


/* ==========================================================
   COMPONENT ANALYSIS
   ========================================================== */

async function selectComponent(
    componentId
) {

    const panel =
        $("#componentPanel");

    if (!panel) {
        return;
    }

    if (!componentId) {

        currentComponent = null;

        panel.innerHTML =
            `
            <div class="empty-state">
                Select a component to view its AI screening result.
            </div>
            `;

        return;
    }

    panel.innerHTML =
        `
        <div class="empty-state">
            Running AI screening for ${escapeHtml(componentId)}...
        </div>
        `;

    try {

        const data =
            await apiGet(
                `/components/${encodeURIComponent(componentId)}`
            );

        currentComponent =
            data;

        renderComponent(
            data
        );

    } catch (error) {

        console.error(
            "Component analysis failed:",
            error
        );

        panel.innerHTML =
            `
            <div class="empty-state">
                Analysis failed:
                ${escapeHtml(error.message)}
            </div>
            `;
    }
}


/* ==========================================================
   COMPONENT RENDER
   ========================================================== */

function renderComponent(
    data
) {

    const panel =
        $("#componentPanel");

    if (!panel) {
        return;
    }

    const risk =
        riskClass(
            data.risk_level
        );

    const unit =
        data.unit || "";

    panel.innerHTML =
        `
        <div class="component-layout">

            <div>

                <div class="component-summary">

                    <div class="component-id">
                        ${escapeHtml(
                            data.component_id
                        )}
                    </div>

                    <div class="component-meta">
                        ${escapeHtml(
                            data.component_type
                        )}
                        •
                        ${escapeHtml(
                            data.parameter_name
                        )}
                    </div>

                    <div class="component-tags">

                        <span class="tag">
                            Lot ${escapeHtml(
                                data.lot_id
                            )}
                        </span>

                        <span class="tag">
                            ${formatNumber(
                                data.temperature_c,
                                1
                            )} °C
                        </span>

                        <span class="tag">
                            ${escapeHtml(unit)}
                        </span>

                    </div>


                    <div class="risk-result">

                        <div class="risk-result-label">
                            AI Risk Decision
                        </div>

                        <div class="risk-result-score">
                            ${formatNumber(
                                data.risk_score,
                                1
                            )}

                            <span
                                style="
                                    font-size:15px;
                                    color:#94a3b8;
                                "
                            >
                                / 100
                            </span>
                        </div>

                        <div class="risk-badge ${risk}">
                            ${escapeHtml(
                                data.risk_level
                            )}
                        </div>

                    </div>


                    <div class="measurement-grid">

                        ${measurementCard(
                            "0h",
                            data.value_0h,
                            unit
                        )}

                        ${measurementCard(
                            "24h",
                            data.value_24h,
                            unit
                        )}

                        ${measurementCard(
                            "96h",
                            data.value_96h,
                            unit
                        )}

                        ${measurementCard(
                            "168h",
                            data.value_168h,
                            unit
                        )}

                    </div>

                </div>

            </div>


            <div>

                ${renderChart(data)}

                ${renderFactors(data)}

                ${renderForecast(data)}

                ${renderExplanation(data)}

            </div>

        </div>
        `;
}


/* ==========================================================
   MEASUREMENT CARD
   ========================================================== */

function measurementCard(
    label,
    value,
    unit
) {

    return `
        <div class="measurement">

            <div class="measurement-label">
                ${escapeHtml(label)}
            </div>

            <div class="measurement-value">

                ${formatNumber(
                    value,
                    3
                )}

                <span class="measurement-unit">
                    ${escapeHtml(unit)}
                </span>

            </div>

        </div>
    `;
}


/* ==========================================================
   RISK FACTORS
   ========================================================== */

function renderFactors(
    data
) {

    return `
        <div class="factors">

            <div class="factors-title">
                Risk Factor Breakdown
            </div>

            ${factorBar(
                "Anomaly Index",
                data.anomaly_index
            )}

            ${factorBar(
                "Early Drift Index",
                data.early_drift_index
            )}

            ${factorBar(
                "Future Drift Index",
                data.future_drift_index
            )}

            ${factorBar(
                "Limit Utilization",
                Math.min(
                    Number(
                        data.limit_utilization || 0
                    ),
                    100
                )
            )}

        </div>
    `;
}


function factorBar(
    label,
    value
) {

    const number =
        Math.max(
            0,
            Math.min(
                100,
                Number(value || 0)
            )
        );

    return `
        <div class="factor">

            <div class="factor-row">

                <span class="factor-label">
                    ${escapeHtml(label)}
                </span>

                <span class="factor-value">
                    ${formatNumber(
                        number,
                        1
                    )}
                </span>

            </div>

            <div class="factor-track">

                <div
                    class="factor-fill"
                    style="width:${number}%"
                ></div>

            </div>

        </div>
    `;
}


/* ==========================================================
   FORECAST
   ========================================================== */

function renderForecast(
    data
) {

    return `
        <div class="explanation">

            <div class="explanation-title">
                AI Forecast
            </div>

            <div class="reason">

                <span class="reason-mark">
                    →
                </span>

                <span>
                    Predicted 168h:
                    <strong>
                        ${formatNumber(
                            data.predicted_168h,
                            3
                        )}
                        ${escapeHtml(
                            data.unit || ""
                        )}
                    </strong>
                </span>

            </div>

            <div class="reason">

                <span class="reason-mark">
                    ↑
                </span>

                <span>
                    Predicted slope:
                    <strong>
                        ${formatNumber(
                            data.predicted_slope,
                            5
                        )}
                    </strong>
                </span>

            </div>

            <div class="reason">

                <span class="reason-mark">
                    ≈
                </span>

                <span>
                    Safety slope:
                    <strong>
                        ${formatNumber(
                            data.safety_slope,
                            5
                        )}
                    </strong>
                </span>

            </div>

            <div class="reason">

                <span class="reason-mark">
                    %
                </span>

                <span>
                    Engineering-limit utilization:
                    <strong>
                        ${formatNumber(
                            data.limit_utilization,
                            1
                        )}%
                    </strong>
                </span>

            </div>

        </div>
    `;
}


/* ==========================================================
   CHART
   ========================================================== */

function renderChart(
    data
) {

    const width = 760;
    const height = 330;

    const left = 60;
    const right = 30;
    const top = 30;
    const bottom = 52;

    const innerWidth =
        width - left - right;

    const innerHeight =
        height - top - bottom;


    const points = [
        {
            label: "0h",
            value: Number(
                data.value_0h
            )
        },

        {
            label: "24h",
            value: Number(
                data.value_24h
            )
        },

        {
            label: "96h",
            value: Number(
                data.value_96h
            )
        },

        {
            label: "168h",
            value: Number(
                data.value_168h
            )
        }
    ];


    const validValues =
        points
            .map(
                point =>
                    point.value
            )
            .filter(
                value =>
                    Number.isFinite(value)
            );


    const predicted =
        Number(
            data.predicted_168h
        );


    const allValues =
        [
            ...validValues,

            ...(Number.isFinite(
                predicted
            )
                ? [predicted]
                : [])
        ];


    if (!allValues.length) {

        return `
            <div class="chart-card">

                <div class="chart-title">
                    Burn-In Trajectory
                </div>

                <div class="chart-subtitle">
                    No measurement data available
                </div>

            </div>
        `;
    }


    let minValue =
        Math.min(
            ...allValues
        );

    let maxValue =
        Math.max(
            ...allValues
        );


    if (
        minValue === maxValue
    ) {

        minValue -= 1;
        maxValue += 1;
    }


    const range =
        maxValue - minValue;

    minValue -=
        range * 0.15;

    maxValue +=
        range * 0.15;


    function x(index) {

        return (
            left
            +
            (
                index / 3
            )
            *
            innerWidth
        );
    }


    function y(value) {

        return (
            top
            +
            (
                1
                -
                (
                    (value - minValue)
                    /
                    (maxValue - minValue)
                )
            )
            *
            innerHeight
        );
    }


    let grid = "";


    for (
        let i = 0;
        i <= 4;
        i++
    ) {

        const yPos =
            top
            +
            (
                i / 4
            )
            *
            innerHeight;


        const yValue =
            maxValue
            -
            (
                i / 4
            )
            *
            (
                maxValue - minValue
            );


        grid +=
            `
            <line
                x1="${left}"
                y1="${yPos}"
                x2="${width - right}"
                y2="${yPos}"
                stroke="#e8edf3"
                stroke-width="1"
            />

            <text
                x="${left - 10}"
                y="${yPos + 4}"
                text-anchor="end"
                font-size="10"
                fill="#94a3b8"
            >
                ${formatNumber(
                    yValue,
                    2
                )}
            </text>
            `;
    }


    const labels = [
        "0h",
        "24h",
        "96h",
        "168h"
    ];


    let xLabels = "";


    labels.forEach(
        (
            label,
            index
        ) => {

            xLabels +=
                `
                <text
                    x="${x(index)}"
                    y="${height - 18}"
                    text-anchor="middle"
                    font-size="10"
                    fill="#94a3b8"
                >
                    ${label}
                </text>
                `;
        }
    );


    const validPoints =
        points
            .map(
                (
                    point,
                    index
                ) => {

                    if (
                        !Number.isFinite(
                            point.value
                        )
                    ) {

                        return null;
                    }

                    return {
                        x:
                            x(index),

                        y:
                            y(
                                point.value
                            ),

                        value:
                            point.value
                    };
                }
            )
            .filter(
                Boolean
            );


    const polyline =
        validPoints
            .map(
                point =>
                    `${point.x},${point.y}`
            )
            .join(" ");


    let circles = "";


    validPoints.forEach(
        point => {

            circles +=
                `
                <circle
                    cx="${point.x}"
                    cy="${point.y}"
                    r="5"
                    fill="#2563eb"
                />

                <text
                    x="${point.x}"
                    y="${point.y - 12}"
                    text-anchor="middle"
                    font-size="10"
                    font-weight="700"
                    fill="#172033"
                >
                    ${formatNumber(
                        point.value,
                        2
                    )}
                </text>
                `;
        }
    );


    let prediction = "";


    if (
        Number.isFinite(
            predicted
        )
    ) {

        const startingPoint =
            validPoints.length
                ? validPoints[
                    validPoints.length - 1
                ]
                : {
                    x: x(1),
                    y: y(
                        Number(
                            data.value_24h
                        )
                    )
                };


        prediction =
            `
            <line
                x1="${startingPoint.x}"
                y1="${startingPoint.y}"
                x2="${x(3)}"
                y2="${y(predicted)}"
                stroke="#dc2626"
                stroke-width="3"
                stroke-dasharray="8 7"
            />

            <circle
                cx="${x(3)}"
                cy="${y(predicted)}"
                r="6"
                fill="#dc2626"
            />

            <text
                x="${x(3) - 8}"
                y="${y(predicted) - 14}"
                text-anchor="end"
                font-size="10"
                font-weight="800"
                fill="#dc2626"
            >
                AI ${formatNumber(
                    predicted,
                    2
                )}
            </text>
            `;
    }


    return `
        <div class="chart-card">

            <div class="chart-title">
                Burn-In Trajectory
            </div>

            <div class="chart-subtitle">
                Measured values and AI-predicted 168h value
            </div>

            <div class="chart">

                <svg
                    viewBox="0 0 ${width} ${height}"
                    preserveAspectRatio="none"
                >

                    ${grid}

                    <polyline
                        points="${polyline}"
                        fill="none"
                        stroke="#2563eb"
                        stroke-width="3"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                    />

                    ${prediction}

                    ${circles}

                    ${xLabels}

                </svg>

            </div>

        </div>
    `;
}


/* ==========================================================
   EXPLANATION
   ========================================================== */

function renderExplanation(
    data
) {

    const reasons =
        Array.isArray(
            data.reasons
        )
            ? data.reasons
            : [];


    let html =
        `
        <div class="explanation">

            <div class="explanation-title">
                Why the AI made this decision
            </div>
        `;


    if (!reasons.length) {

        html +=
            `
            <div class="reason">

                <span class="reason-mark">
                    ✓
                </span>

                <span>
                    No major risk indicators were triggered.
                </span>

            </div>
            `;

    } else {

        for (
            const reason
            of reasons
        ) {

            html +=
                `
                <div class="reason">

                    <span class="reason-mark">
                        •
                    </span>

                    <span>
                        ${escapeHtml(
                            reason
                        )}
                    </span>

                </div>
                `;
        }
    }


    html +=
        `
        </div>
        `;

    return html;
}


/* ==========================================================
   TABLE
   ========================================================== */

function renderInitialTable() {

    const body =
        $("#componentTableBody");

    if (!body) {
        return;
    }

    const rows = [];

    for (
        let i = 1;
        i <= 20;
        i++
    ) {

        const id =
            `CMP-${String(i).padStart(6, "0")}`;

        rows.push(
            `
            <tr data-component-id="${id}">

                <td class="component-cell">
                    ${id}
                </td>

                <td>
                    —
                </td>

                <td class="parameter-cell">
                    Select to analyze
                </td>

                <td>
                    —
                </td>

                <td>
                    —
                </td>

                <td>
                    —
                </td>

                <td>
                    <span class="table-risk normal">
                        View
                    </span>
                </td>

            </tr>
            `
        );
    }

    body.innerHTML =
        rows.join("");


    body
        .querySelectorAll(
            "tr[data-component-id]"
        )
        .forEach(
            row => {

                row.addEventListener(
                    "click",
                    () => {

                        const id =
                            row.dataset.componentId;

                        const select =
                            $("#componentSelect");

                        if (select) {

                            select.value =
                                id;
                        }

                        selectComponent(
                            id
                        );
                    }
                );
            }
        );
}


/* ==========================================================
   REFRESH
   ========================================================== */

async function refreshDashboard() {

    const button =
        $("#refreshButton");

    if (button) {
        button.classList.add(
            "loading"
        );
    }

    try {

        await loadHealth();

        await loadMetadata();

    } catch (error) {

        console.error(
            "Refresh failed:",
            error
        );

    } finally {

        if (button) {

            button.classList.remove(
                "loading"
            );
        }
    }
}


/* ==========================================================
   EVENTS
   ========================================================== */

const componentSelect =
    $("#componentSelect");

if (componentSelect) {

    componentSelect.addEventListener(
        "change",
        event => {

            selectComponent(
                event.target.value
            );
        }
    );
}


const refreshButton =
    $("#refreshButton");

if (refreshButton) {

    refreshButton.addEventListener(
        "click",
        refreshDashboard
    );
}


/* ==========================================================
   INITIALIZATION
   ========================================================== */

async function init() {

    const status =
        $("#systemStatus");

    if (status) {

        status.textContent =
            "Connecting...";
    }


    try {

        await loadHealth();

        await loadMetadata();

        buildComponentSelector();

        renderInitialTable();

        /*
         * Automatically analyze CMP-000001 so that the
         * dashboard immediately shows real AI output.
         */

        const select =
            $("#componentSelect");

        if (select) {

            select.value =
                "CMP-000001";
        }

        await selectComponent(
            "CMP-000001"
        );

    } catch (error) {

        console.error(
            "Dashboard initialization failed:",
            error
        );

        if (status) {

            status.textContent =
                "Backend connection failed";
        }
    }
}


/* ==========================================================
   START
   ========================================================== */

init();