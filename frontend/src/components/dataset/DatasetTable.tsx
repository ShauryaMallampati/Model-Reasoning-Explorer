import React from "react";

const DatasetTable: React.FC<{ examples: Record<string, string>[] }> = ({ examples }) => {
  if (examples.length === 0) {
    return <div className="panel">No examples yet</div>;
  }
  const headers = Object.keys(examples[0]);
  return (
    <div className="panel">
      <h3>Examples</h3>
      <table className="table">
        <thead>
          <tr>
            {headers.map((h) => (
              <th key={h}>{h}</th>
            ))}
            <th>Run</th>
          </tr>
        </thead>
        <tbody>
          {examples.map((row, idx) => (
            <tr key={idx}>
              {headers.map((h) => (
                <td key={h}>{row[h]}</td>
              ))}
              <td>{row.run_id ? <a href={`/runs/${row.run_id}`}>Open</a> : "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default DatasetTable;
