class RiskEngine:
    """
    AegisBurn AI risk engine.

    IMPORTANT:
    This engine performs inference ONLY from measured burn-in data,
    engineered anomaly features, and the model's prediction.

    Ground-truth dataset fields such as:
        - defect_type
        - is_defective

    are deliberately NOT used here.

    They may exist in a labeled evaluation dataset, but they must
    never influence a live screening decision.
    """

    def __init__(self):
        pass

    @staticmethod
    def _float(value, default=0.0):
        try:
            value = float(value)
            if value != value:
                return default
            return value
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value, low=0.0, high=100.0):
        return max(low, min(high, value))

    def evaluate_component(
        self,
        row,
        predicted_168h,
        safety_slope
    ):
        """
        Calculate component risk using ONLY inference-time information.

        Inputs used:
            anomaly_flag
            anomaly_score
            anomaly_index
            burn-in measurements
            engineered drift values
            predicted 168h value
            safety slope
            engineering limit

        Ground-truth labels are ignored.
        """

        # ==========================================================
        # BASIC INPUTS
        # ==========================================================

        anomaly_flag = bool(
            row.get("anomaly_flag", False)
        )

        anomaly_score = self._float(
            row.get(
                "anomaly_score",
                row.get("anomaly_index", 0.0)
            )
        )

        anomaly_index = self._float(
            row.get(
                "anomaly_index",
                anomaly_score
            )
        )

        predicted_168h = self._float(
            predicted_168h
        )

        safety_slope = self._float(
            safety_slope
        )

        value_0h = self._float(
            row.get("value_0h")
        )

        value_24h = self._float(
            row.get("value_24h")
        )

        value_96h = self._float(
            row.get("value_96h")
        )

        value_168h = self._float(
            row.get("value_168h")
        )

        engineering_limit = self._float(
            row.get("engineering_limit"),
            0.0
        )

        # ==========================================================
        # DERIVED DRIFT VALUES
        # ==========================================================

        drift_0_24 = self._float(
            row.get(
                "drift_0_24",
                value_24h - value_0h
            )
        )

        drift_24_96 = self._float(
            row.get(
                "drift_24_96",
                value_96h - value_24h
            )
        )

        # NOTE: These are ALWAYS computed from predicted_168h, NOT
        # read from the row even if a drift_96_168/drift_0_168
        # column already exists there. main.py's create_features()
        # precomputes these columns from the ACTUAL value_168h when
        # it is present (e.g. an uploaded CSV that includes real
        # 168h data), which would leak ground truth straight into
        # risk_score via total_relative_drift below. This is a
        # predictive risk score computed before 168h is known, so
        # the real future value must never reach it — only the
        # model's own prediction may.
        drift_96_168 = (
            predicted_168h - value_96h
        )

        drift_0_168 = (
            predicted_168h - value_0h
        )

        early_slope = self._float(
            row.get(
                "slope_early",
                drift_0_24 / 24.0
            )
        )

        # ==========================================================
        # PREDICTED FUTURE DRIFT
        # ==========================================================

        future_drift = predicted_168h - value_24h

        if future_drift < 0:
            future_drift_for_risk = 0.0
        else:
            future_drift_for_risk = future_drift

        # ==========================================================
        # LIMIT UTILIZATION
        # ==========================================================

        limit_utilization = 0.0

        if engineering_limit > 0:

            # NOTE: value_168h is intentionally EXCLUDED here.
            # This is a predictive risk score computed BEFORE
            # 168h is known — including the actual future value
            # would leak ground truth into the risk decision and
            # defeat the entire point of early screening. Only
            # values available during real early-stage screening
            # (0h/24h/96h) plus the model's own prediction are
            # used, even if an uploaded CSV happens to include an
            # actual value_168h (e.g. for retrospective/evaluation
            # datasets).
            observed_max = max(
                value_0h,
                value_24h,
                value_96h,
                predicted_168h
            )

            limit_utilization = (
                observed_max /
                engineering_limit
            ) * 100.0

            limit_utilization = self._clamp(
                limit_utilization
            )

        # ==========================================================
        # RISK SCORE
        # ==========================================================

        risk_score = 0.0
        reasons = []

        # ----------------------------------------------------------
        # 1. AI ANOMALY DETECTION
        # ----------------------------------------------------------

        if anomaly_flag:
            risk_score += 45.0

            reasons.append(
                "AI anomaly detector identified abnormal burn-in behavior."
            )

            anomaly_extra = (
                self._clamp(anomaly_index) *
                0.30
            )

            risk_score += anomaly_extra

        elif anomaly_index >= 70:
            risk_score += 30.0

            reasons.append(
                "AI anomaly score indicates elevated abnormality."
            )

        elif anomaly_index >= 50:
            risk_score += 15.0

            reasons.append(
                "AI anomaly score indicates moderate abnormality."
            )

        # ----------------------------------------------------------
        # 1b. LOT-RELATIVE DEVIATION
        #
        # A component can look unremarkable against the global
        # population but be a clear outlier within its own lot
        # (or vice versa). Surface this explicitly for QA.
        # ----------------------------------------------------------

        lot_worst_hour = row.get(
            "lot_worst_hour"
        )

        lot_worst_zscore = row.get(
            "lot_worst_zscore"
        )

        if (
            lot_worst_hour is not None
            and lot_worst_zscore is not None
        ):

            try:
                lot_worst_zscore = float(
                    lot_worst_zscore
                )
            except (TypeError, ValueError):
                lot_worst_zscore = 0.0

            if abs(lot_worst_zscore) >= 2.0:

                lot_worst_value = row.get(
                    "lot_worst_value"
                )

                lot_worst_mean = row.get(
                    "lot_worst_mean"
                )

                direction = (
                    "above"
                    if lot_worst_zscore > 0
                    else "below"
                )

                reasons.append(
                    f"At {lot_worst_hour}, this component reads "
                    f"{lot_worst_value:.3f} vs. its own lot's "
                    f"average of {lot_worst_mean:.3f} "
                    f"({abs(lot_worst_zscore):.1f}sigma {direction} "
                    "lot baseline)."
                    if lot_worst_value is not None
                    and lot_worst_mean is not None
                    else (
                        f"At {lot_worst_hour}, this component "
                        f"deviates {abs(lot_worst_zscore):.1f}sigma "
                        f"{direction} its own lot's typical value."
                    )
                )

        # ----------------------------------------------------------
        # 2. EARLY DRIFT
        # ----------------------------------------------------------

        if value_0h != 0:

            early_relative_drift = (
                abs(drift_0_24) /
                abs(value_0h)
            ) * 100.0

        else:
            early_relative_drift = 0.0

        if early_relative_drift >= 20.0:

            risk_score += 20.0

            reasons.append(
                "Large parameter drift detected during the first 24 hours."
            )

        elif early_relative_drift >= 10.0:

            risk_score += 12.0

            reasons.append(
                "Elevated parameter drift detected during the first 24 hours."
            )

        elif early_relative_drift >= 5.0:

            risk_score += 6.0

            reasons.append(
                "Moderate early parameter drift detected."
            )

        # ----------------------------------------------------------
        # 3. LONG-TERM DRIFT
        # ----------------------------------------------------------

        if value_0h != 0:

            total_relative_drift = (
                abs(drift_0_168) /
                abs(value_0h)
            ) * 100.0

        else:
            total_relative_drift = 0.0

        if total_relative_drift >= 50.0:

            risk_score += 20.0

            reasons.append(
                "Severe parameter drift observed across the burn-in period."
            )

        elif total_relative_drift >= 30.0:

            risk_score += 14.0

            reasons.append(
                "Significant parameter drift observed across the burn-in period."
            )

        elif total_relative_drift >= 15.0:

            risk_score += 7.0

            reasons.append(
                "Moderate cumulative parameter drift detected."
            )

        # ----------------------------------------------------------
        # 4. FUTURE PREDICTION
        # ----------------------------------------------------------

        if value_24h != 0:

            predicted_relative_change = (
                future_drift_for_risk /
                abs(value_24h)
            ) * 100.0

        else:
            predicted_relative_change = 0.0

        if predicted_relative_change >= 50.0:

            risk_score += 15.0

            reasons.append(
                "Model predicts severe future parameter degradation."
            )

        elif predicted_relative_change >= 25.0:

            risk_score += 10.0

            reasons.append(
                "Model predicts significant future parameter degradation."
            )

        elif predicted_relative_change >= 10.0:

            risk_score += 5.0

            reasons.append(
                "Model predicts measurable future parameter drift."
            )

        # ----------------------------------------------------------
        # 5. ENGINEERING LIMIT
        # ----------------------------------------------------------

        if engineering_limit > 0:

            if limit_utilization >= 100.0:

                risk_score += 20.0

                reasons.append(
                    "Measured or predicted parameter exceeds the engineering limit."
                )

            elif limit_utilization >= 85.0:

                risk_score += 15.0

                reasons.append(
                    "Parameter is approaching the engineering limit."
                )

            elif limit_utilization >= 70.0:

                risk_score += 8.0

                reasons.append(
                    "Parameter utilization of the engineering limit is elevated."
                )

        # ==========================================================
        # 6. SAFETY SLOPE
        # ==========================================================

        # Positive slope means the parameter is increasing.
        # For the prototype parameter families, increasing values
        # represent degradation.

        if safety_slope > 0:

            if value_0h != 0:

                slope_relative = (
                    abs(safety_slope) /
                    abs(value_0h)
                ) * 100.0

            else:
                slope_relative = 0.0

            if slope_relative >= 1.0:

                risk_score += 10.0

                reasons.append(
                    "Positive degradation slope indicates continuing parameter growth."
                )

            elif slope_relative >= 0.25:

                risk_score += 5.0

                reasons.append(
                    "Burn-in measurements show a continuing degradation trend."
                )

        # ==========================================================
        # FINAL SCORE
        # ==========================================================

        risk_score = self._clamp(
            risk_score
        )

        # ==========================================================
        # RISK LEVEL
        # ==========================================================

        if risk_score >= 80:

            risk_level = "CRITICAL"

        elif risk_score >= 60:

            risk_level = "HIGH"

        elif risk_score >= 30:

            risk_level = "MEDIUM"

        else:

            risk_level = "LOW"

        # ==========================================================
        # DECISION
        # ==========================================================

        if risk_level == "CRITICAL":

            recommendation = (
                "FLAG COMPONENT — severe degradation or anomaly indicators "
                "detected. Immediate inspection recommended."
            )

        elif risk_level == "HIGH":

            recommendation = (
                "FLAG COMPONENT — significant abnormal behavior detected. "
                "Additional inspection recommended."
            )

        elif risk_level == "MEDIUM":

            recommendation = (
                "REVIEW COMPONENT — abnormal drift indicators detected. "
                "Continue monitoring and review."
            )

        else:

            recommendation = (
                "PASS / MONITOR — no significant abnormal behavior detected "
                "from the available burn-in measurements."
            )

        # ==========================================================
        # FALLBACK EXPLANATION
        # ==========================================================

        if not reasons:

            reasons.append(
                "No significant anomaly, drift, prediction, or limit indicators detected."
            )

        # Remove duplicate reasons while preserving order.

        reasons = list(
            dict.fromkeys(reasons)
        )

        # ==========================================================
        # BEHAVIOR PATTERN CLASSIFICATION
        #
        # Names WHICH failure pattern the trajectory shape
        # resembles -- not just a risk number. This is purely
        # shape-based from features already computed above; no
        # ground-truth labels are used. Directly targets the
        # explainability requirement: a QA inspector sees "this
        # looks like a Latent Defect" instead of just "risk: 62".
        # ==========================================================

        early_index = self._clamp(
            early_relative_drift
        )

        future_index = self._clamp(
            predicted_relative_change
        )

        # Did the value spike early then partially recover, rather
        # than keep climbing? A real recovery-after-spike shape,
        # not just "some drift happened between 24h and 96h".
        is_spike_recovery = (
            drift_0_24 > 0
            and drift_24_96 < 0
            and abs(drift_24_96)
            >= abs(drift_0_24) * 0.3
        )

        if not anomaly_flag:

            if future_index >= 25:

                behavior_pattern = "WATCH"

                behavior_pattern_label = (
                    "Potential Latent Defect (Early Warning)"
                )

                behavior_pattern_explanation = (
                    "Not yet flagged as anomalous, but the "
                    "predicted 168h trajectory shows meaningful "
                    "drift relative to the engineering limit. "
                    "Worth monitoring -- this is exactly the "
                    "early-warning case a static pass/fail limit "
                    "would miss entirely."
                )

            else:

                behavior_pattern = "NORMAL"

                behavior_pattern_label = "Normal"

                behavior_pattern_explanation = (
                    "Early and predicted future drift are both "
                    "within the expected range for this "
                    "component family."
                )

        elif is_spike_recovery:

            behavior_pattern = "SUDDEN_ANOMALY"

            behavior_pattern_label = "Sudden Anomaly (Transient Spike)"

            behavior_pattern_explanation = (
                "A sharp early spike followed by partial recovery "
                "-- the value moved a lot in one interval, then "
                "pulled back. This transient shape is "
                "characteristic of a sudden anomaly rather than "
                "steady degradation."
            )

        elif early_index < 15 and future_index >= 40:

            behavior_pattern = "LATENT_DEFECT"

            behavior_pattern_label = "Latent Defect"

            behavior_pattern_explanation = (
                "This is the defining latent-defect signature: "
                "early behavior (0h-24h) looked unremarkable -- "
                "it would have passed a static early-screening "
                "limit -- but the predicted trajectory shows "
                "substantial drift developing later. A dynamic, "
                "trajectory-aware model is what catches this; a "
                "fixed pass/fail threshold at 24h would not."
            )

        else:

            behavior_pattern = "GRADUAL_DRIFT"

            behavior_pattern_label = "Gradual Drift"

            behavior_pattern_explanation = (
                "Drift is elevated and escalating steadily across "
                "the burn-in period, rather than appearing as a "
                "single spike or staying hidden until late."
            )

        # ==========================================================
        # RETURN
        # ==========================================================

        return {
            "risk_score": round(
                risk_score,
                2
            ),

            "risk_level": risk_level,

            "anomaly_flag": anomaly_flag,

            "anomaly_score": round(
                anomaly_score,
                4
            ),

            "anomaly_index": round(
                self._clamp(anomaly_index),
                2
            ),

            "predicted_168h": round(
                predicted_168h,
                4
            ),

            "safety_slope": round(
                safety_slope,
                6
            ),

            "early_drift": round(
                drift_0_24,
                4
            ),

            "future_drift": round(
                future_drift,
                4
            ),

            "early_drift_index": round(
                early_index,
                2
            ),

            "future_drift_index": round(
                future_index,
                2
            ),

            "limit_utilization": round(
                limit_utilization,
                2
            ),

            "engineering_limit": round(
                engineering_limit,
                4
            ),

            "behavior_pattern": behavior_pattern,

            "behavior_pattern_label": behavior_pattern_label,

            "behavior_pattern_explanation":
                behavior_pattern_explanation,

            "risk_reasons": reasons,

            "reasons": reasons,

            "risk_decision": risk_level,

            "recommendation": recommendation,
        }