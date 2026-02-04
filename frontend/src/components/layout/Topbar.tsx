import React from "react";

const Topbar: React.FC = () => {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <span className="topbar-logo">Model Reasoning Explorer</span>
      </div>
      <div className="topbar-right">
        <span className="topbar-badge">Inference Debugger</span>
      </div>
    </header>
  );
};

export default Topbar;
