import React from "react";
import { StructuredCaseText } from "../structuredCaseText/structuredCaseText";
import "./itemizedDecision.scss";

export const ItemizedDecision = ({ decision }) => {
  if (!decision?.decision && !decision?.shortAnswer && !decision?.reasons?.length) return null;

  return (
    <div className="itemized_decision">
      <div className="headline">
        <span className="label">Decision:</span>{" "}
        <span className="value">{decision?.decision || "—"}</span>
      </div>

      {decision?.shortAnswer ? (
        <div className="short">
          <StructuredCaseText text={decision.shortAnswer} />
        </div>
      ) : null}

      {Array.isArray(decision?.reasons) && decision.reasons.length > 0 ? (
        <ul className="reasons">
          {decision.reasons.slice(0, 6).map((r, idx) => (
            <li key={idx}>{r}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
};


