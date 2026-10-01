import React from "react";

type AttributionTextProps = {
  tokens: string[];
  scores: number[];
};

const AttributionText: React.FC<AttributionTextProps> = ({ tokens, scores }) => {
  if (tokens.length !== scores.length || scores.some((score) => !Number.isFinite(score))) {
    return <p role="alert">Attribution values do not match the recorded model tokens.</p>;
  }
  const max = Math.max(...scores.map((s) => Math.abs(s)), Number.EPSILON);
  return (
    <div className="token-grid">
      {tokens.map((token, idx) => {
        const score = scores[idx] ?? 0;
        const intensity = Math.min(1, Math.abs(score) / max);
        const color = score >= 0 ? `rgba(255, 124, 0, ${intensity})` : `rgba(0, 148, 255, ${intensity})`;
        return (
          <span key={idx} className="token" title={`Attribution: ${score.toPrecision(5)}`}
            style={{ background: color, whiteSpace: "pre-wrap" }}>
            {token}
          </span>
        );
      })}
    </div>
  );
};

export default AttributionText;
