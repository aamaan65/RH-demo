import React from "react";
import "./structuredCaseText.scss";

const renderInlineBold = (text) => {
  const s = String(text ?? "");
  if (!s.includes("**")) return s;
  const parts = s.split("**");
  const out = [];
  for (let i = 0; i < parts.length; i++) {
    const chunk = parts[i];
    if (!chunk) continue;
    if (i % 2 === 1) {
      out.push(
        <strong key={`b-${i}`} className="sct_inline_bold">
          {chunk}
        </strong>
      );
    } else {
      out.push(<React.Fragment key={`t-${i}`}>{chunk}</React.Fragment>);
    }
  }
  return out;
};

export const StructuredCaseText = ({ text = "", className = "" }) => {
  const raw = String(text ?? "");
  if (!raw.trim()) return null;

  const lines = raw.replace(/\r\n/g, "\n").split("\n");
  const blocks = [];
  let paraLines = [];
  let listItems = [];

  const flushPara = () => {
    if (paraLines.length === 0) return;
    const t = paraLines.join("\n").trimEnd();
    if (t) {
      blocks.push(
        <div key={`p-${blocks.length}`} className="sct_paragraph">
          {t.split("\n").map((ln, idx) => (
            <React.Fragment key={idx}>
              {idx > 0 ? <br /> : null}
              {renderInlineBold(ln)}
            </React.Fragment>
          ))}
        </div>
      );
    }
    paraLines = [];
  };

  const flushList = () => {
    if (listItems.length === 0) return;
    blocks.push(
      <ul key={`ul-${blocks.length}`} className="sct_bullets">
        {listItems.map((li, idx) => (
          <li key={idx}>{renderInlineBold(li)}</li>
        ))}
      </ul>
    );
    listItems = [];
  };

  for (const line of lines) {
    const trimmed = line.trim();
    const isBullet = /^[-•*]\s+/.test(trimmed);
    if (isBullet) {
      flushPara();
      listItems.push(trimmed.replace(/^[-•*]\s+/, ""));
      continue;
    }
    if (trimmed === "") {
      flushPara();
      flushList();
      continue;
    }
    flushList();
    paraLines.push(line);
  }

  flushPara();
  flushList();

  if (blocks.length === 0) return <span>{raw}</span>;
  return <div className={`structured_case_text ${className || ""}`}>{blocks}</div>;
};


