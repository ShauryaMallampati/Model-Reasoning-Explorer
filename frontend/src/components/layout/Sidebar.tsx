import React from "react";
import { NavLink } from "react-router-dom";

const Sidebar: React.FC = () => {
  return (
    <aside className="sidebar">
      <div className="sidebar-title">MRE</div>
      <nav className="sidebar-nav">
        <NavLink to="/" className={({ isActive }) => (isActive ? "active" : "")}
          >Run Launcher</NavLink
        >
        <NavLink to="/compare" className={({ isActive }) => (isActive ? "active" : "")}
          >Compare</NavLink
        >
        <NavLink to="/dataset" className={({ isActive }) => (isActive ? "active" : "")}
          >Dataset Explorer</NavLink
        >
      </nav>
    </aside>
  );
};

export default Sidebar;
