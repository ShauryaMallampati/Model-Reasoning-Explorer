import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import Home from "./pages/Home";
import RunViewer from "./pages/RunViewer";
import CompareViewer from "./pages/CompareViewer";
import DatasetExplorer from "./pages/DatasetExplorer";
import Sidebar from "./components/layout/Sidebar";
import Topbar from "./components/layout/Topbar";
import ErrorBoundary from "./components/common/ErrorBoundary";

const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <div className="app-shell">
        <Topbar />
        <div className="app-body">
          <Sidebar />
          <main className="app-main">
            <Routes>
              <Route path="/" element={<Home />} />
              <Route path="/runs/:runId" element={<RunViewer />} />
              <Route path="/compare" element={<CompareViewer />} />
              <Route path="/dataset" element={<DatasetExplorer />} />
              <Route path="*" element={<Navigate to="/" />} />
            </Routes>
          </main>
        </div>
      </div>
    </ErrorBoundary>
  );
};

export default App;
